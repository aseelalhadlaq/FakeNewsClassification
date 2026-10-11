"""
multifc_experiment.py

Wires MultiFCPreprocessor (data), DAMiniLMModel (model, class 2 of the
LIAR2 three-class set), and DAMiniLMExperimentRunner (orchestration,
class 3) together for MultiFC specifically -- this is the other half of
this project's central claim: whether DA-MiniLM-RWAC's advantage on
LIAR2 REPRODUCES when independently retrained and evaluated on MultiFC.

Per this project's established framing, MultiFC is evaluated as
INDEPENDENT RETRAINING on its own official train/dev/test split -- NOT as
zero-shot transfer from a LIAR2-trained model. Concretely: DA-MiniLM-RWAC
and its comparators are retrained from scratch on MultiFC's train split,
model-selected on MultiFC's dev split, and evaluated on MultiFC's test
split. There is no chronological condition here (MultiFC has no role in
this project's temporal-drift analysis) -- only the official split.

No modeling logic is reimplemented here. DAMiniLMModel and
DAMiniLMExperimentRunner are reused directly, just configured with
num_classes=3 and MultiFC's label order -- confirmed to work correctly
after fixing a real bug (evaluate_predictions() was silently defaulting
to num_classes=6 for Brier score/NLL, which breaks on any non-6-class
label space via a shape mismatch; now explicit).

Usage
-----
    from multifc_preprocessor import MultiFCPreprocessor
    from da_minilm_model import DAMiniLMModel
    from da_minilm_experiment_runner import DAMiniLMExperimentRunner
    from multifc_experiment import MultiFCExperiment

    prep = MultiFCPreprocessor()
    df = prep.run_pipeline(train_path, dev_path, test_path)

    model = DAMiniLMModel(output_dir="./multifc_results", num_classes=3)
    runner = DAMiniLMExperimentRunner(splitter=None, model=model, num_classes=3)

    exp = MultiFCExperiment(preprocessor=prep, model=model, runner=runner)

    embeddings = {mode: model.encode_texts_minilm(df[f"model_text__{mode}"].tolist(), cache_key=f"multifc_{mode}")
                  for mode in ["claim_only", "claim_plus_safe_metadata"]}

    results_df = exp.run_battery(df, embeddings, seeds=[42, 43, 44, 45, 46],
                                  text_modes=["claim_only", "claim_plus_safe_metadata"])

    ablation_df = exp.run_component_ablation(df, embeddings, text_mode="claim_plus_safe_metadata", seed=42)
"""

from typing import Sequence

import numpy as np
import pandas as pd


class MultiFCExperiment:

    def __init__(self, preprocessor, model, runner):
        """`preprocessor` a MultiFCPreprocessor instance (used for
        get_official_split() -- the only "splitting" MultiFC needs).
        `model` a DAMiniLMModel instance configured with num_classes=3.
        `runner` a DAMiniLMExperimentRunner instance, also configured with
        num_classes=3 (splitter=None is fine -- MultiFC never calls it)."""
        self.p = preprocessor
        self.model = model
        self.runner = runner

    def _assemble_arrays(self, df: pd.DataFrame, embeddings: dict, text_mode: str):
        """Builds the single concatenated (X, y, time_rank, train_idx,
        valid_idx, test_idx) arrays run_single_experiment() expects, from
        MultiFCPreprocessor.get_official_split(). time_rank is all zeros
        -- MultiFC has no temporal condition, but the array is required by
        run_single_experiment()'s signature (inherited from the LIAR2
        pipeline) for interface compatibility."""
        train_df, dev_df, test_df = self.p.get_official_split(df)
        # test_df may include officially-unlabelled rows; keep only labelled
        # ones for evaluation (this project has no gold labels for the
        # official unlabelled MultiFC test partition).
        test_df = test_df[test_df["label_id"].notna()].copy()

        X_train = embeddings[text_mode][train_df.index.values]
        X_valid = embeddings[text_mode][dev_df.index.values]
        X_test = embeddings[text_mode][test_df.index.values]

        y_train = train_df["label_id"].values.astype(int)
        y_valid = dev_df["label_id"].values.astype(int)
        y_test = test_df["label_id"].values.astype(int)

        X_all = np.vstack([X_train, X_valid, X_test])
        y_all = np.concatenate([y_train, y_valid, y_test])
        time_rank_all = np.zeros(len(y_all))

        train_idx = np.arange(len(y_train))
        valid_idx = np.arange(len(y_train), len(y_train) + len(y_valid))
        test_idx = np.arange(len(y_train) + len(y_valid), len(y_all))

        return X_all, y_all, time_rank_all, train_idx, valid_idx, test_idx

    def run_battery(
        self, df: pd.DataFrame, embeddings: dict, seeds: Sequence[int],
        text_modes: Sequence[str], model_kinds: Sequence[str] = ("minilm_logreg", "da_minilm_mlp"),
    ):
        """DA-MiniLM-RWAC / MiniLM-LogReg-RW, retrained from scratch on
        MultiFC's official train, model-selected on dev, evaluated on
        test -- across seeds and text modes. No split_mode loop (MultiFC
        has only the one, official, condition)."""
        all_rows = []
        prediction_store = {}

        for text_mode in text_modes:
            X_all, y_all, time_rank_all, train_idx, valid_idx, test_idx = self._assemble_arrays(
                df, embeddings, text_mode,
            )
            for seed in seeds:
                for model_kind in model_kinds:
                    print(f"running [{model_kind}] MultiFC text={text_mode} seed={seed}")
                    row, lambda_df, pred, proba = self.runner.run_single_experiment(
                        X_all, y_all, time_rank_all, train_idx, valid_idx, test_idx,
                        text_mode=text_mode, split_mode="official_independent_retraining",
                        seed=seed, model_kind=model_kind,
                    )
                    all_rows.append(row)
                    prediction_store[(row["model"], text_mode, seed)] = {
                        "y_test": y_all[test_idx], "pred": pred, "proba": proba,
                    }
                    print(f"  macro_f1={row['macro_f1']:.4f}")

        results_df = pd.DataFrame(all_rows)
        return results_df, prediction_store

    def run_component_ablation(self, df: pd.DataFrame, embeddings: dict, text_mode: str, seed: int,
                                calibration_mode: str = "two_stage") -> pd.DataFrame:
        """The same six-cell ablation (Uniform-LR / LogReg+RW / MiniLM-
        LogReg-RW / Uniform-MLP / MLP+RW / DA-MiniLM-RWAC) run on
        MultiFC's official split -- lets you check whether the reweighting
        and calibration contributions found on LIAR2 hold up on an
        independently-retrained second dataset, not just the model's
        headline comparison."""
        X_all, y_all, time_rank_all, train_idx, valid_idx, test_idx = self._assemble_arrays(
            df, embeddings, text_mode,
        )
        return self.runner.run_component_ablation(
            X_all, y_all, time_rank_all, train_idx, valid_idx, test_idx,
            text_mode=text_mode, split_mode="official_independent_retraining", seed=seed,
            calibration_mode=calibration_mode,
        )


if __name__ == "__main__":
    print("multifc_experiment.py loaded. Class: MultiFCExperiment")