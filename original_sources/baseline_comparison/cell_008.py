"""
da_minilm_experiment_runner.py

Class 3 of 3: experiment orchestration. Uses TemporalSplitter (class 1)
for splitting and DAMiniLMModel (class 2) for training/prediction -- this
class owns everything in between and around those two: metrics, sample
weighting (class-balance + recency), calibration (bias + temperature
selection), the recency-lambda grid search, multi-seed batteries,
result summarization, paired bootstrap comparison, and rolling-window
temporal evaluation.

--------------------------------------------------------------------------
A known, deliberately-preserved bug: the calibration temperature grid
--------------------------------------------------------------------------
select_best_calibration() grid-searches (tau, temperature) and keeps
whichever combination gets the best VALIDATION macro-F1. But temperature
scaling divides every class's logit by the same positive scalar -- a
monotonic transform that CANNOT change which class has the highest logit,
i.e. it cannot change argmax, i.e. it cannot change macro-F1 at all. For
any fixed tau, every temperature in temperature_grid produces the exact
same macro-F1. Because the selection loop uses strict `>` comparison,
the FIRST temperature value in the grid for the winning tau is silently
kept -- not because it's better, but because later ties never satisfy
`score > best["score"]`. This matches this project's own documented
finding ("temperature non-identifiable under macro-F1 objective; locks to
first grid value via iteration-order artifact") -- verified directly
(temperature confirmed to have zero effect on macro-F1 across the full
grid, and the first grid value confirmed to silently win). Reproduced
here byte-for-byte, commented rather than silently fixed, per this
project's practice of disclosing known bugs rather than patching them out
after the fact. Temperature DOES still matter for ECE/Brier/NLL (those
depend on the actual probabilities, not just argmax) -- it only fails to
matter for the objective this selection loop actually uses by default.
Pass objective="nll" to select_best_calibration() for an objective
temperature actually affects.
--------------------------------------------------------------------------

Usage
-----
    from temporal_splitter import TemporalSplitter
    from da_minilm_model import DAMiniLMModel
    from da_minilm_experiment_runner import DAMiniLMExperimentRunner

    splitter = TemporalSplitter()
    model = DAMiniLMModel(output_dir="./da_minilm_results")
    runner = DAMiniLMExperimentRunner(splitter=splitter, model=model)

    embeddings = {mode: model.encode_texts_minilm(df[f"model_text__{mode}"].tolist(), cache_key=mode)
                  for mode in ["claim_only", "claim_plus_safe_metadata"]}

    main_results_df, lambda_selection_df, prediction_store = runner.run_full_battery(
        df, embeddings, seeds=[42, 43, 44, 45, 46],
        text_modes=["claim_only", "claim_plus_safe_metadata"],
        split_modes=["chronological", "random_stratified"],
    )

    summary_df = runner.summarize_results(main_results_df, group_cols=["model", "text_mode", "split_mode"])
    bootstrap_df = runner.run_paired_bootstrap(prediction_store,
                                                text_modes=["claim_only", "claim_plus_safe_metadata"],
                                                split_modes=["chronological", "random_stratified"],
                                                seeds=[42, 43, 44, 45, 46])
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, log_loss,
)


NUM_CLASSES_DEFAULT = 6


# ============================================================
# Metrics
# ============================================================

def ordinal_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def within_one_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Multiclass ECE based on confidence of the predicted class."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def multiclass_brier_score(y_true, proba, num_classes: int = NUM_CLASSES_DEFAULT) -> float:
    y_true = np.asarray(y_true)
    y_onehot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def safe_nll(y_true, proba, num_classes: int = NUM_CLASSES_DEFAULT) -> float:
    return float(log_loss(y_true, np.clip(proba, 1e-12, 1.0), labels=list(range(num_classes))))


def evaluate_predictions(
    y_true, y_pred, proba=None, model_name="model", split_mode=None, text_mode=None,
    seed=None, recency_lambda=None, tau=None, temperature=None, extra=None,
    num_classes: int = NUM_CLASSES_DEFAULT,
) -> dict:
    """`num_classes` MUST match proba.shape[1] -- it defaults to 6 (LIAR2)
    for backward compatibility, but this default silently breaks on any
    other label space (e.g. MultiFC's 3 classes) via a shape mismatch in
    multiclass_brier_score/safe_nll. Always pass num_classes explicitly
    when working outside the 6-class LIAR2/LIAR label space -- confirmed
    this default was a real bug when first reusing this code for MultiFC."""
    row = {
        "model": model_name, "split_mode": split_mode, "text_mode": text_mode, "seed": seed,
        "recency_lambda": recency_lambda, "tau": tau, "temperature": temperature,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "within_one_accuracy": within_one_accuracy(y_true, y_pred),
    }
    if proba is not None:
        row.update({
            "ece": expected_calibration_error(y_true, proba),
            "brier": multiclass_brier_score(y_true, proba, num_classes=num_classes),
            "nll": safe_nll(y_true, proba, num_classes=num_classes),
        })
    if extra:
        row.update(extra)
    return row


# ============================================================
# Sample weighting
# ============================================================

def normalize_weights(w, min_weight: float = 0.05, max_weight: float = 20.0) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, min_weight, max_weight)
    return w / (np.mean(w) + 1e-12)


def effective_number_class_weights(y, beta: float = 0.999, num_classes: int = NUM_CLASSES_DEFAULT) -> np.ndarray:
    """Per-class weights based on the effective number of samples (Cui et al., 2019)."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=num_classes).astype(float)
    weights = np.ones(num_classes, dtype=float)
    for c in range(num_classes):
        n = counts[c]
        if n > 0:
            effective_num = (1.0 - beta ** n) / (1.0 - beta)
            weights[c] = 1.0 / effective_num
    return weights / (weights.mean() + 1e-12)


def class_sample_weights(y_train, beta: float = 0.999, num_classes: int = NUM_CLASSES_DEFAULT) -> np.ndarray:
    class_w = effective_number_class_weights(y_train, beta=beta, num_classes=num_classes)
    return normalize_weights(class_w[np.asarray(y_train, dtype=int)])


def recency_sample_weights(time_rank_train, lam: float = 0.0) -> np.ndarray:
    """Exponential recency weighting: age = max_train_time - sample_time;
    higher lambda gives more weight to recent training samples."""
    t = np.asarray(time_rank_train, dtype=float)
    age = t.max() - t
    w = np.exp(-float(lam) * age * 10.0)
    return normalize_weights(w)


def combined_sample_weights(y_train, time_rank_train, recency_lambda: float = 0.0,
                             beta: float = 0.999, num_classes: int = NUM_CLASSES_DEFAULT) -> np.ndarray:
    w_class = class_sample_weights(y_train, beta=beta, num_classes=num_classes)
    w_recency = recency_sample_weights(time_rank_train, recency_lambda)
    return normalize_weights(w_class * w_recency)


# ============================================================
# Calibration
# ============================================================

def softmax_np(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    return expv / (expv.sum(axis=1, keepdims=True) + 1e-12)


def proba_to_logits(proba, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(proba, eps, 1.0))


def logit_adjustment_bias(y_train, num_classes: int = NUM_CLASSES_DEFAULT, tau: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    prior = counts / counts.sum()
    return tau * np.log(prior + eps)


def apply_bias_and_temperature(proba, bias=None, temperature: float = 1.0) -> np.ndarray:
    logits = proba_to_logits(proba)
    if bias is not None:
        logits = logits - bias.reshape(1, -1)
    logits = logits / max(float(temperature), 1e-6)
    return softmax_np(logits)


def select_best_calibration(
    valid_proba, y_valid, y_train, objective: str = "macro_f1",
    tau_grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    temperature_grid: Sequence[float] = (0.75, 1.0, 1.25, 1.5, 2.0),
    num_classes: int = NUM_CLASSES_DEFAULT,
) -> dict:
    """Select tau and temperature using validation only.

    NOTE: with objective="macro_f1" (the default), temperature has no
    effect on the selected argmax and is therefore effectively
    unidentifiable -- see module docstring. Use objective="nll" for an
    objective temperature actually affects, or use
    select_best_calibration_two_stage() below instead of this function.
    """
    best = {"tau": 0.0, "temperature": 1.0, "score": -1e9}
    for tau in tau_grid:
        bias = logit_adjustment_bias(y_train, tau=tau, num_classes=num_classes)
        for temp in temperature_grid:
            calibrated = apply_bias_and_temperature(valid_proba, bias=bias, temperature=temp)
            pred = calibrated.argmax(axis=1)
            if objective == "macro_f1":
                score = f1_score(y_valid, pred, average="macro", zero_division=0)
            elif objective == "nll":
                score = -safe_nll(y_valid, calibrated, num_classes=num_classes)
            else:
                raise ValueError(f"Unknown objective: {objective}")
            if score > best["score"]:
                best = {"tau": tau, "temperature": temp, "score": float(score)}
    return best


def select_best_calibration_two_stage(
    valid_proba, y_valid, y_train,
    tau_grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    temperature_grid: Sequence[float] = (0.75, 1.0, 1.25, 1.5, 2.0),
    num_classes: int = NUM_CLASSES_DEFAULT,
) -> dict:
    """The calibration-bug fix. select_best_calibration() jointly grid-
    searches (tau, temperature) under a single objective -- but temperature
    has zero effect on argmax, so under objective="macro_f1" it can never
    be meaningfully selected (see module docstring; confirmed by direct
    test). This function decouples the two selections onto the objective
    each one actually affects:

    Stage 1 -- select tau under macro_f1 (tau/logit-adjustment DOES shift
               the decision boundary via the class-prior bias, so this
               objective is meaningful here).
    Stage 2 -- with tau fixed at its stage-1 value, select temperature
               under NLL (temperature doesn't move argmax but does reshape
               the probability distribution, so NLL -- which depends on
               the full distribution, not just the top class -- is
               sensitive to it. This is the "objective each dimension can
               actually see" pairing that was missing from the joint
               search.)

    Returns {"tau", "temperature", "macro_f1_at_best_tau", "nll_at_best_temperature"}.
    """
    # Stage 1: tau, evaluated at temperature=1.0 (irrelevant to macro_f1 anyway).
    best_tau, best_tau_score = 0.0, -1e9
    for tau in tau_grid:
        bias = logit_adjustment_bias(y_train, tau=tau, num_classes=num_classes)
        calibrated = apply_bias_and_temperature(valid_proba, bias=bias, temperature=1.0)
        score = f1_score(y_valid, calibrated.argmax(axis=1), average="macro", zero_division=0)
        if score > best_tau_score:
            best_tau, best_tau_score = tau, float(score)

    # Stage 2: temperature, with tau fixed, evaluated under NLL.
    bias = logit_adjustment_bias(y_train, tau=best_tau, num_classes=num_classes)
    best_temp, best_temp_score = 1.0, 1e9
    for temp in temperature_grid:
        calibrated = apply_bias_and_temperature(valid_proba, bias=bias, temperature=temp)
        nll = safe_nll(y_valid, calibrated, num_classes=num_classes)
        if nll < best_temp_score:
            best_temp, best_temp_score = temp, float(nll)

    return {
        "tau": best_tau, "temperature": best_temp,
        "macro_f1_at_best_tau": best_tau_score, "nll_at_best_temperature": best_temp_score,
    }


# ============================================================
# Statistical comparison
# ============================================================

def paired_bootstrap_macro_f1_diff(y_true, y_pred_a, y_pred_b, n_boot: int = 2000, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    y_true, y_pred_a, y_pred_b = np.asarray(y_true), np.asarray(y_pred_a), np.asarray(y_pred_b)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        f1_a = f1_score(y_true[idx], y_pred_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], y_pred_b[idx], average="macro", zero_division=0)
        diffs[b] = f1_a - f1_b

    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    prop_le_0 = float(np.mean(diffs <= 0))
    prop_ge_0 = float(np.mean(diffs >= 0))
    p_value = min(2.0 * min(prop_le_0, prop_ge_0), 1.0)

    return {
        "mean_diff": float(diffs.mean()), "ci_lower_95": float(ci_lower), "ci_upper_95": float(ci_upper),
        "p_value": p_value, "significant_at_95": bool(ci_lower > 0 or ci_upper < 0),
    }


# ============================================================
# Experiment runner
# ============================================================

class DAMiniLMExperimentRunner:

    def __init__(
        self,
        splitter,
        model,
        recency_lambdas: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30),
        class_weight_beta: float = 0.999,
        temperature_grid: Sequence[float] = (0.75, 1.0, 1.25, 1.5, 2.0),
        tau_grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
        num_classes: int = NUM_CLASSES_DEFAULT,
        output_dir=None,
    ):
        """`splitter` should be a TemporalSplitter instance (class 1) --
        used for chronological_split / random_stratified_split /
        make_rolling_windows. `model` should be a DAMiniLMModel instance
        (class 2) -- used for train_da_minilm_mlp / train_logreg_baseline.
        Neither is reimplemented here."""
        self.splitter = splitter
        self.model = model
        self.recency_lambdas = recency_lambdas
        self.class_weight_beta = class_weight_beta
        self.temperature_grid = temperature_grid
        self.tau_grid = tau_grid
        self.num_classes = num_classes
        self.output_dir = output_dir or getattr(model, "output_dir", None)

    # ---------------- Single-run orchestration ----------------

    def run_single_experiment(
        self, X, y, time_rank, train_idx, valid_idx, test_idx,
        text_mode: str, split_mode: str, seed: int, model_kind: str = "da_minilm_mlp",
        use_reweighting: bool = True, use_calibration: bool = True,
        calibration_mode: str = "joint_macro_f1", model_name_override: Optional[str] = None,
    ):
        """model_kind: "da_minilm_mlp" (MLP head) or "minilm_logreg"
        (Logistic Regression head) -- the architecture dimension of the
        component ablation.

        use_reweighting: if False, sample weights are uniform (no class
        balancing, no recency weighting) and self.recency_lambdas is
        skipped entirely -- this is the ablation's "Uniform" condition
        for whichever architecture is selected (e.g. model_kind=
        "minilm_logreg" + use_reweighting=False == Uniform-LR).

        use_calibration: if False, raw (uncalibrated) argmax predictions
        are used directly -- no tau/temperature search at all. This
        isolates the calibration step's contribution.

        calibration_mode: only relevant when use_calibration=True.
            "joint_macro_f1" -- the ORIGINAL notebook's approach: joint
                (tau, temperature) grid search under macro_f1. Preserved
                for exact reproduction; see module docstring re: the
                temperature non-identifiability this causes.
            "two_stage" -- the BUG FIX: tau selected under macro_f1,
                temperature separately selected under NLL. Use this for
                new results; use "joint_macro_f1" only to reproduce prior
                numbers exactly.

        Returns (row, lambda_df, pred_test, test_cal). `lambda_df` has a
        single row (recency_lambda=0.0) when use_reweighting=False, since
        there is no lambda grid to search in that case.
        """
        X_train, X_valid, X_test = X[train_idx], X[valid_idx], X[test_idx]
        y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
        t_train = time_rank[train_idx]

        recency_lambdas = self.recency_lambdas if use_reweighting else (0.0,)

        lambda_candidates = []
        stored = {}

        for rec_lam in recency_lambdas:
            if use_reweighting:
                w_train = combined_sample_weights(
                    y_train, t_train, recency_lambda=rec_lam,
                    beta=self.class_weight_beta, num_classes=self.num_classes,
                )
            else:
                w_train = np.ones(len(y_train), dtype=float)

            if model_kind == "da_minilm_mlp":
                valid_proba, test_proba, raw_valid_f1 = self.model.train_da_minilm_mlp(
                    X_train, y_train, w_train, X_valid, y_valid, X_test, seed=seed,
                )
                default_model_name = "DA-MiniLM-RWAC" if use_reweighting else "Uniform-MLP"
            elif model_kind == "minilm_logreg":
                valid_proba, test_proba = self.model.train_logreg_baseline(
                    X_train, y_train, w_train, X_valid, y_valid, X_test, seed=seed,
                )
                raw_valid_f1 = f1_score(y_valid, valid_proba.argmax(axis=1), average="macro", zero_division=0)
                default_model_name = "MiniLM-LogReg-RW" if use_reweighting else "Uniform-LR"
            else:
                raise ValueError(f"Unknown model_kind: {model_kind}")
            model_name = model_name_override or default_model_name

            if use_calibration:
                if calibration_mode == "joint_macro_f1":
                    calib = select_best_calibration(
                        valid_proba, y_valid, y_train, objective="macro_f1",
                        tau_grid=self.tau_grid, temperature_grid=self.temperature_grid, num_classes=self.num_classes,
                    )
                elif calibration_mode == "two_stage":
                    calib_raw = select_best_calibration_two_stage(
                        valid_proba, y_valid, y_train,
                        tau_grid=self.tau_grid, temperature_grid=self.temperature_grid, num_classes=self.num_classes,
                    )
                    calib = {"tau": calib_raw["tau"], "temperature": calib_raw["temperature"]}
                else:
                    raise ValueError(f"Unknown calibration_mode: {calibration_mode}")
            else:
                calib = {"tau": 0.0, "temperature": 1.0}

            valid_cal = apply_bias_and_temperature(
                valid_proba,
                bias=logit_adjustment_bias(y_train, tau=calib["tau"], num_classes=self.num_classes),
                temperature=calib["temperature"],
            )
            valid_macro_f1 = f1_score(y_valid, valid_cal.argmax(axis=1), average="macro", zero_division=0)

            lambda_candidates.append({
                "recency_lambda": rec_lam, "valid_macro_f1": valid_macro_f1,
                "raw_valid_macro_f1": raw_valid_f1, "tau": calib["tau"], "temperature": calib["temperature"],
            })
            stored[rec_lam] = (valid_proba, test_proba, calib)

        lambda_df = pd.DataFrame(lambda_candidates).sort_values("valid_macro_f1", ascending=False)
        best_lambda = float(lambda_df.iloc[0]["recency_lambda"])
        valid_proba, test_proba, calib = stored[best_lambda]

        bias = logit_adjustment_bias(y_train, tau=calib["tau"], num_classes=self.num_classes)
        test_cal = apply_bias_and_temperature(test_proba, bias=bias, temperature=calib["temperature"])
        pred_test = test_cal.argmax(axis=1)

        row = evaluate_predictions(
            y_test, pred_test, proba=test_cal, model_name=model_name, split_mode=split_mode,
            text_mode=text_mode, seed=seed, recency_lambda=best_lambda, tau=calib["tau"],
            temperature=calib["temperature"], num_classes=self.num_classes,
            extra={
                "n_train": len(train_idx), "n_valid": len(valid_idx), "n_test": len(test_idx),
                "selected_valid_macro_f1": float(lambda_df.iloc[0]["valid_macro_f1"]),
                "use_reweighting": use_reweighting, "use_calibration": use_calibration,
                "calibration_mode": calibration_mode if use_calibration else None,
            },
        )
        return row, lambda_df, pred_test, test_cal

    # ---------------- Component ablation ----------------

    def run_component_ablation(
        self, X, y, time_rank, train_idx, valid_idx, test_idx,
        text_mode: str, split_mode: str, seed: int, calibration_mode: str = "two_stage",
    ) -> pd.DataFrame:
        """The four ablation cells isolating each design choice's
        contribution, plus the two full models for reference:

            Uniform-LR         : LogReg, no reweighting, no calibration
            LogReg + RW         : LogReg, reweighting, no calibration
            MiniLM-LogReg-RW    : LogReg, reweighting, calibration (full)
            Uniform-MLP         : MLP,    no reweighting, no calibration
            MLP + RW            : MLP,    reweighting, no calibration
            DA-MiniLM-RWAC      : MLP,    reweighting, calibration (full)

        Reading down each architecture's three rows isolates reweighting's
        contribution (row 2 minus row 1) and calibration's contribution
        (row 3 minus row 2), holding architecture fixed. Reading across
        LogReg vs MLP at matched reweighting/calibration settings isolates
        the architecture's own contribution.
        """
        cells = [
            ("minilm_logreg", False, False, "Uniform-LR"),
            ("minilm_logreg", True, False, "LogReg + RW (no calibration)"),
            ("minilm_logreg", True, True, "MiniLM-LogReg-RW"),
            ("da_minilm_mlp", False, False, "Uniform-MLP"),
            ("da_minilm_mlp", True, False, "MLP + RW (no calibration)"),
            ("da_minilm_mlp", True, True, "DA-MiniLM-RWAC"),
        ]
        rows = []
        for model_kind, use_rw, use_calib, label in cells:
            print(f"running ablation cell: {label}")
            row, _, _, _ = self.run_single_experiment(
                X, y, time_rank, train_idx, valid_idx, test_idx,
                text_mode=text_mode, split_mode=split_mode, seed=seed, model_kind=model_kind,
                use_reweighting=use_rw, use_calibration=use_calib,
                calibration_mode=calibration_mode, model_name_override=label,
            )
            rows.append(row)
            print(f"  macro_f1={row['macro_f1']:.4f}")
        return pd.DataFrame(rows)

    # ---------------- Full battery ----------------

    def run_full_battery(
        self, df: pd.DataFrame, embeddings: dict, seeds: Sequence[int],
        text_modes: Sequence[str], split_modes: Sequence[str],
        model_kinds: Sequence[str] = ("minilm_logreg", "da_minilm_mlp"),
        calibration_mode: str = "two_stage",
    ):
        """`embeddings` must be a dict {text_mode: np.ndarray} (from
        model.encode_texts_minilm()). Splitting is delegated to
        self.splitter -- NOT reimplemented here. `df` must have a reset
        integer index (positions == train_idx/valid_idx/test_idx from the
        split functions), a `label_id` column, and a `time_rank` column.

        BUG FIX: this method previously did not pass calibration_mode to
        run_single_experiment() at all, so every call silently used
        run_single_experiment()'s own default of "joint_macro_f1" -- the
        calibration scheme where temperature has no effect on macro-F1 and
        is therefore never meaningfully selected (see
        select_best_calibration()'s docstring). run_component_ablation()
        already defaulted to the fixed "two_stage" scheme; run_full_battery
        (the method that actually produces your main/headline results) did
        not. Now it does, by default."""
        y_all = df["label_id"].values.astype(int)
        time_rank_all = df["time_rank"].values.astype(float)

        all_rows, all_lambda_rows = [], []
        prediction_store = {}

        for text_mode in text_modes:
            X = embeddings[text_mode]
            for split_mode in split_modes:
                for seed in seeds:
                    train_idx, valid_idx, test_idx = self.splitter.get_split(df, split_mode, seed=seed)

                    for model_kind in model_kinds:
                        print(f"running [{model_kind}] text={text_mode} split={split_mode} seed={seed}")
                        row, lambda_df, pred, proba = self.run_single_experiment(
                            X, y_all, time_rank_all, train_idx, valid_idx, test_idx,
                            text_mode=text_mode, split_mode=split_mode, seed=seed, model_kind=model_kind,
                            calibration_mode=calibration_mode,
                        )
                        all_rows.append(row)
                        for _, r in lambda_df.iterrows():
                            all_lambda_rows.append({
                                "model": row["model"], "text_mode": text_mode, "split_mode": split_mode,
                                "seed": seed, **r.to_dict(),
                            })
                        prediction_store[(row["model"], text_mode, split_mode, seed)] = {
                            "y_test": y_all[test_idx], "pred": pred, "proba": proba,
                        }
                        print(f"  macro_f1={row['macro_f1']:.4f}, ece={row['ece']:.4f}, lambda={row['recency_lambda']}")

        main_results_df = pd.DataFrame(all_rows)
        lambda_selection_df = pd.DataFrame(all_lambda_rows)

        if self.output_dir is not None:
            main_results_df.to_csv(f"{self.output_dir}/main_da_minilm_results.csv", index=False)
            lambda_selection_df.to_csv(f"{self.output_dir}/lambda_selection_validation_results.csv", index=False)

        return main_results_df, lambda_selection_df, prediction_store

    # ---------------- Summaries and comparisons ----------------

    def summarize_results(self, results_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
        metrics = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "ece", "brier", "nll"]
        agg = {}
        for m in metrics:
            if m in results_df.columns:
                agg[f"{m}_mean"] = (m, "mean")
                agg[f"{m}_std"] = (m, "std")
                agg[f"{m}_min"] = (m, "min")
                agg[f"{m}_max"] = (m, "max")
        return results_df.groupby(group_cols).agg(**agg, n_runs=("macro_f1", "count")).reset_index()

    def run_paired_bootstrap(
        self, prediction_store: dict, text_modes: Sequence[str], split_modes: Sequence[str],
        seeds: Sequence[int], model_a: str = "DA-MiniLM-RWAC", model_b: str = "MiniLM-LogReg-RW",
        n_boot: int = 2000,
    ) -> pd.DataFrame:
        rows = []
        for text_mode in text_modes:
            for split_mode in split_modes:
                for seed in seeds:
                    key_a = (model_a, text_mode, split_mode, seed)
                    key_b = (model_b, text_mode, split_mode, seed)
                    if key_a not in prediction_store or key_b not in prediction_store:
                        continue
                    y_test = prediction_store[key_a]["y_test"]
                    boot = paired_bootstrap_macro_f1_diff(
                        y_test, prediction_store[key_a]["pred"], prediction_store[key_b]["pred"],
                        n_boot=n_boot, seed=seed,
                    )
                    rows.append({
                        "comparison": f"{model_a} minus {model_b}", "text_mode": text_mode,
                        "split_mode": split_mode, "seed": seed, **boot,
                    })
        bootstrap_df = pd.DataFrame(rows)
        if self.output_dir is not None:
            bootstrap_df.to_csv(f"{self.output_dir}/paired_bootstrap_results.csv", index=False)
        return bootstrap_df

    def run_rolling_window_evaluation(
        self, df: pd.DataFrame, embeddings: dict, text_mode: str, seeds: Sequence[int],
        model_kind: str = "da_minilm_mlp", rolling_windows_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Rolling windows are built by self.splitter.make_rolling_windows()
        -- not reimplemented here."""
        rolling_windows_kwargs = rolling_windows_kwargs or {}
        windows = self.splitter.make_rolling_windows(df, **rolling_windows_kwargs)

        y_all = df["label_id"].values.astype(int)
        time_rank_all = df["time_rank"].values.astype(float)
        X = embeddings[text_mode]

        rows = []
        for w in windows:
            for seed in seeds:
                print(f"running [rolling] window={w['window_id']} seed={seed}")
                row, lambda_df, pred, proba = self.run_single_experiment(
                    X, y_all, time_rank_all, w["train_idx"], w["valid_idx"], w["test_idx"],
                    text_mode=text_mode, split_mode="rolling_window", seed=seed, model_kind=model_kind,
                )
                row.update({
                    "window_id": w["window_id"], "train_end_frac": w["train_end_frac"],
                    "valid_end_frac": w["valid_end_frac"], "test_end_frac": w["test_end_frac"],
                })
                rows.append(row)
                print(f"  macro_f1={row['macro_f1']:.4f}, ece={row['ece']:.4f}, lambda={row['recency_lambda']}")

        rolling_results_df = pd.DataFrame(rows)
        if self.output_dir is not None:
            rolling_results_df.to_csv(f"{self.output_dir}/rolling_window_results.csv", index=False)
        return rolling_results_df


if __name__ == "__main__":
    print("da_minilm_experiment_runner.py loaded. Class: DAMiniLMExperimentRunner")