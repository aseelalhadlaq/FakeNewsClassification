"""
run_all.py

Complete, ordered, end-to-end run of every experiment built in this
project. Paste this into a Kaggle notebook (as one script cell, or split
at the "# ===== SECTION" markers into separate cells) AFTER pasting in
every class file this script imports from:

    liar_preprocessor.py           liar2_preprocessor.py
    baseline_models.py             multifc_preprocessor.py
    temporal_splitter.py           da_minilm_model.py
    da_minilm_experiment_runner.py multifc_experiment.py
    shuffle_control.py             drift_analysis.py

Requires internet enabled on the Kaggle notebook (Settings -> Internet ->
On) for: DA-MiniLM-RWAC/MiniLM-LogReg-RW MiniLM embeddings, DistilBERT
fine-tuning, and MultiFC MiniLM embeddings. A GPU accelerator is strongly
recommended for the MLP/DistilBERT sections.

Update the four path variables in SECTION 0 to match your actual Kaggle
dataset paths before running.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# SECTION 0: Paths -- EDIT THESE to match your Kaggle input structure
# ============================================================

LIAR_TRAIN = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar dataset/liar dataset/train.tsv"
LIAR_VALID = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar dataset/liar dataset/valid.tsv"
LIAR_TEST = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar dataset/liar dataset/test.tsv"

LIAR2_TRAIN = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/train.csv"
LIAR2_VALID = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/valid.csv"
LIAR2_TEST = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/test.csv"

# Verify these three against your actual MultiFC dataset folder listing
# before running -- filenames/casing vary by upload (e.g. train.tsv vs
# train.tsv.txt). List the folder first if unsure:
#     for p in sorted(Path(MULTIFC_FOLDER).rglob("*")): print(p)
MULTIFC_TRAIN = "/kaggle/input/datasets/badalahmmed/multifc/train.tsv"
MULTIFC_DEV = "/kaggle/input/datasets/badalahmmed/multifc/dev.tsv"
MULTIFC_TEST = "/kaggle/input/datasets/badalahmmed/multifc/test.tsv"

OUTPUT_ROOT = Path("/kaggle/working/full_run_results")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
TEXT_MODES = ["claim_only", "claim_plus_safe_metadata"]

print("=" * 70)
print("SECTION 0 complete: paths configured, output root:", OUTPUT_ROOT)
print("=" * 70)


# ============================================================
# SECTION 1: Load and preprocess LIAR and LIAR2
# ============================================================

#from liar_preprocessor import LiarPreprocessor
#from liar2_preprocessor import Liar2Preprocessor

liar_prep = LiarPreprocessor()
liar_df = liar_prep.run_pipeline(LIAR_TRAIN, LIAR_VALID, LIAR_TEST)

liar2_prep = Liar2Preprocessor()
liar2_df = liar2_prep.run_pipeline(LIAR2_TRAIN, LIAR2_VALID, LIAR2_TEST)

print("=" * 70)
print(f"SECTION 1 complete: LIAR {len(liar_df)} rows, LIAR2 {len(liar2_df)} rows")
print("=" * 70)


# ============================================================
# SECTION 2: Classical + transformer baselines (LIAR, LIAR2, merge,
# chronological, cross-dataset)
# ============================================================

#from baseline_models import BaselineModels

baselines = BaselineModels(liar_preprocessor=liar_prep, liar2_preprocessor=liar2_prep)

# Set to True only if this notebook has internet + GPU enabled.
INCLUDE_TRANSFORMER_BASELINES = True

baseline_results_df = baselines.run_all_experiments(
    liar_df, liar2_df, include_transformer_baselines=INCLUDE_TRANSFORMER_BASELINES,
)
baseline_results_df.to_csv(OUTPUT_ROOT / "baseline_results.csv", index=False)

print("=" * 70)
print(f"SECTION 2 complete: {len(baseline_results_df)} baseline rows saved.")
print("=" * 70)


# ============================================================
# SECTION 3: DistilBERT chronological, multi-seed (a previously
# outstanding required experiment -- only in the baseline battery
# above under the OFFICIAL split, not chronological)
# ============================================================

if INCLUDE_TRANSFORMER_BASELINES:
    distilbert_chrono_df, distilbert_chrono_summary = baselines.run_distilbert_multiseed(
        liar2_df, "LIAR2 chronological", seeds=SEEDS, split_mode="chronological",
    )
    distilbert_chrono_df.to_csv(OUTPUT_ROOT / "distilbert_chronological_multiseed.csv", index=False)
    print("DistilBERT chronological summary:", distilbert_chrono_summary)
else:
    print("Skipping DistilBERT-chronological (INCLUDE_TRANSFORMER_BASELINES=False).")

print("=" * 70)
print("SECTION 3 complete.")
print("=" * 70)


# ============================================================
# SECTION 4: LIAR2 deduplicated + chronologically split cohort
# (used by BOTH the drift diagnostics and the DA-MiniLM experiments,
# matching the notebook this project's central claim is built on)
# ============================================================

audit = liar2_prep.audit_duplicate_statements(liar2_df)
liar2_dedup_df = liar2_prep.drop_duplicate_statements(liar2_df, policy="keep_first", audit_df=audit)
liar2_dedup_df = liar2_dedup_df.reset_index(drop=True)

chrono_train_idx, chrono_valid_idx, chrono_test_idx = liar2_prep.chronological_split(liar2_dedup_df)
drift_train_df = liar2_dedup_df.iloc[chrono_train_idx].copy()
drift_test_df = liar2_dedup_df.iloc[chrono_test_idx].copy()

print(f"Deduplicated chronological cohort: "
      f"train={len(chrono_train_idx)}, valid={len(chrono_valid_idx)}, test={len(chrono_test_idx)}")
print("=" * 70)
print("SECTION 4 complete.")
print("=" * 70)


# ============================================================
# SECTION 5: Drift diagnostics (D01-D05) on the deduplicated
# chronological cohort
# ============================================================

#from drift_analysis import DriftAnalysis

da = DriftAnalysis()

INCLUDE_EMBEDDING_SHIFT = INCLUDE_TRANSFORMER_BASELINES  # D04 needs Hugging Face Hub too

drift_summary_df = da.run_all_drift_experiments(
    drift_train_df, drift_test_df, include_embedding_shift=INCLUDE_EMBEDDING_SHIFT,
)
drift_summary_df.to_csv(OUTPUT_ROOT / "drift_summary.csv", index=False)

drift_comparison_df = da.compare_to_reported(drift_summary_df)
drift_comparison_df.to_csv(OUTPUT_ROOT / "drift_comparison_vs_reported.csv", index=False)

print("=" * 70)
print("SECTION 5 complete: drift diagnostics saved.")
print("=" * 70)


# ============================================================
# SECTION 6: DA-MiniLM-RWAC / MiniLM-LogReg-RW -- the main model
# battery, on the FULL (non-deduplicated) LIAR2 pipeline output,
# matching da-minilm-main-model-v1-0.ipynb exactly (that notebook does
# not deduplicate before its own chronological/random splits).
# ============================================================

#from temporal_splitter import TemporalSplitter
#from da_minilm_model import DAMiniLMModel
#from da_minilm_experiment_runner import DAMiniLMExperimentRunner

liar2_full_df = liar2_df.reset_index(drop=True)
liar2_full_df = liar2_prep.add_time_rank(liar2_full_df)

da_model = DAMiniLMModel(output_dir=str(OUTPUT_ROOT / "da_minilm"), num_classes=6)
da_runner = DAMiniLMExperimentRunner(splitter=liar2_prep, model=da_model, num_classes=6)

# Embeddings, once per text mode (cached to disk automatically by DAMiniLMModel).
da_embeddings = {
    mode: da_model.encode_texts_minilm(liar2_full_df[f"model_text__{mode}"].tolist(), cache_key=mode)
    for mode in TEXT_MODES
}

main_results_df, lambda_selection_df, prediction_store = da_runner.run_full_battery(
    liar2_full_df, da_embeddings, seeds=SEEDS, text_modes=TEXT_MODES,
    split_modes=["chronological", "random_stratified"],
)
main_results_df.to_csv(OUTPUT_ROOT / "da_minilm_main_results.csv", index=False)
lambda_selection_df.to_csv(OUTPUT_ROOT / "da_minilm_lambda_selection.csv", index=False)

summary_df = da_runner.summarize_results(main_results_df, group_cols=["model", "text_mode", "split_mode"])
summary_df.to_csv(OUTPUT_ROOT / "da_minilm_summary.csv", index=False)

print("=" * 70)
print(f"SECTION 6 complete: {len(main_results_df)} DA-MiniLM main-battery rows saved.")
print("=" * 70)


# ============================================================
# SECTION 7: Paired bootstrap -- the statistical test for the central
# claim (DA-MiniLM-RWAC vs MiniLM-LogReg-RW)
# ============================================================

bootstrap_df = da_runner.run_paired_bootstrap(
    prediction_store, text_modes=TEXT_MODES, split_modes=["chronological", "random_stratified"],
    seeds=SEEDS, model_a="DA-MiniLM-RWAC", model_b="MiniLM-LogReg-RW", n_boot=2000,
)
bootstrap_df.to_csv(OUTPUT_ROOT / "da_minilm_paired_bootstrap.csv", index=False)

print("Paired bootstrap results:")
print(bootstrap_df[["text_mode", "split_mode", "seed", "mean_diff", "p_value", "significant_at_95"]])
print("=" * 70)
print("SECTION 7 complete.")
print("=" * 70)


# ============================================================
# SECTION 8: Component ablation -- isolates reweighting's and
# calibration's individual contributions, on the condition carrying
# the central claim (chronological + claim_plus_safe_metadata)
# ============================================================

ABLATION_TEXT_MODE = "claim_plus_safe_metadata"
ABLATION_SPLIT_MODE = "chronological"
ABLATION_SEED = SEEDS[0]

ablation_train_idx, ablation_valid_idx, ablation_test_idx = liar2_prep.chronological_split(liar2_full_df)

ablation_df = da_runner.run_component_ablation(
    da_embeddings[ABLATION_TEXT_MODE],
    liar2_full_df["label_id"].values.astype(int),
    liar2_full_df["time_rank"].values.astype(float),
    ablation_train_idx, ablation_valid_idx, ablation_test_idx,
    text_mode=ABLATION_TEXT_MODE, split_mode=ABLATION_SPLIT_MODE, seed=ABLATION_SEED,
    calibration_mode="two_stage",
)
ablation_df.to_csv(OUTPUT_ROOT / "da_minilm_component_ablation.csv", index=False)

print("Component ablation:")
print(ablation_df[["model", "macro_f1", "use_reweighting", "use_calibration"]])
print("=" * 70)
print("SECTION 8 complete.")
print("=" * 70)


# ============================================================
# SECTION 9: Shuffle-control validation (label permutation null test)
# -- uses the fast LogReg head, 300 shuffles, on the same condition as
# the ablation above
# ============================================================

#from shuffle_control import ShuffleControlValidator

X_shuf = da_embeddings[ABLATION_TEXT_MODE]
y_shuf = liar2_full_df["label_id"].values.astype(int)

X_train_s, y_train_s = X_shuf[ablation_train_idx], y_shuf[ablation_train_idx]
X_valid_s, y_valid_s = X_shuf[ablation_valid_idx], y_shuf[ablation_valid_idx]
X_test_s, y_test_s = X_shuf[ablation_test_idx], y_shuf[ablation_test_idx]


def _shuffle_train_predict_fn(X_tr, y_tr, X_va, y_va, X_te, seed):
    w = np.ones(len(y_tr))
    return da_model.train_logreg_baseline(X_tr, y_tr, w, X_va, y_va, X_te, seed=seed)


shuffle_validator = ShuffleControlValidator(_shuffle_train_predict_fn, n_shuffles=300, seed=42)
shuffle_result = shuffle_validator.run(X_train_s, y_train_s, X_valid_s, y_valid_s, X_test_s, y_test_s)
shuffle_validator.print_summary(shuffle_result)

pd.DataFrame([shuffle_result["summary"]]).to_csv(OUTPUT_ROOT / "shuffle_control_summary.csv", index=False)
np.save(OUTPUT_ROOT / "shuffle_control_null_scores.npy", shuffle_result["null_scores"])

print("=" * 70)
print("SECTION 9 complete.")
print("=" * 70)


# ============================================================
# SECTION 10: Rolling-window temporal evaluation
# ============================================================

rolling_results_df = da_runner.run_rolling_window_evaluation(
    liar2_full_df, da_embeddings, text_mode="claim_plus_safe_metadata",
    seeds=SEEDS, model_kind="da_minilm_mlp",
)
rolling_results_df.to_csv(OUTPUT_ROOT / "da_minilm_rolling_window.csv", index=False)

print("=" * 70)
print(f"SECTION 10 complete: {len(rolling_results_df)} rolling-window rows saved.")
print("=" * 70)


# ============================================================
# SECTION 11: SKIPPED for now (MultiFC's official test split has
# zero labelled rows -- crashes StandardScaler/predict_proba on an
# empty array; needs a dev-set split fix before re-enabling).
# ============================================================
# ============================================================
# SECTION 12: Final summary
# ============================================================

print()
print("=" * 70)
print("FULL RUN COMPLETE. All outputs saved under:", OUTPUT_ROOT)
print("=" * 70)
for f in sorted(OUTPUT_ROOT.rglob("*.csv")):
    print(" ", f.relative_to(OUTPUT_ROOT))