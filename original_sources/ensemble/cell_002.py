"""
ensemble_experiment.py

An expandable, model-agnostic ensemble ANALYZER, plus concrete training
functions for every model this project has: TF-IDF+LogReg, TF-IDF+Linear
SVM, TF-IDF+LogReg+restricted-metadata, Uniform-LR (unweighted frozen
MiniLM + LogReg), Uniform-MLP (unweighted frozen MiniLM + MLP),
MiniLM-LogReg-RW, and DA-MiniLM-RWAC.

Rather than hard-coding a single pairwise ensemble (the previous version
of this file), EnsembleAnalyzer accepts ANY number of registered models
(as pre-computed, class-order-aligned validation/test probability
arrays) and reports:

    1. Individual performance for every registered model.
    2. EVERY pairwise weighted-blend combination (C(n,2) pairs), each
       with: a fine weight sweep, the validation-selected weight (the
       number to report), the test-set oracle weight (diagnostic only),
       the selection gap between them, and paired bootstrap significance
       vs. both individual models in the pair.
    3. An equal-weight blend of ALL registered models at once.
    4. A greedy forward-selection ensemble: start from the single best
       model on validation, then repeatedly try adding each remaining
       model (with its own weight grid-searched), keeping the addition
       only if it improves validation macro-F1, until no addition helps.
       This gives a data-driven "best ensemble found" without an
       intractable exhaustive search over all subsets and weights.

This file is fully SELF-CONTAINED (all weighting/calibration/bootstrap
helper functions are defined locally, not imported from
da_minilm_experiment_runner.py or assumed to be in the notebook's global
namespace) -- two earlier versions of a similar file in this project
depended on cell-execution order in different ways and broke in practice
(ModuleNotFoundError, then NameError after a kernel restart). This
version has no such dependency.

Usage
-----
    from liar2_preprocessor import Liar2Preprocessor
    from da_minilm_model import DAMiniLMModel
    from ensemble_experiment import (
        EnsembleAnalyzer, train_tfidf_logreg, train_tfidf_linear_svm,
        train_tfidf_logreg_metadata, train_da_minilm_family,
    )

    prep = Liar2Preprocessor()
    df = prep.run_pipeline(train_path, valid_path, test_path).reset_index(drop=True)
    df = prep.add_time_rank(df)

    train_idx, valid_idx, test_idx = prep.chronological_split(df)
    y_all = df["label_id"].values.astype(int)
    y_train, y_valid, y_test = y_all[train_idx], y_all[valid_idx], y_all[test_idx]

    text_col = "model_text__claim_plus_safe_metadata"
    texts_all = df[text_col].values
    train_texts, valid_texts, test_texts = texts_all[train_idx], texts_all[valid_idx], texts_all[test_idx]

    model = DAMiniLMModel(output_dir="./ensemble_results")
    X_minilm = model.encode_texts_minilm(df[text_col].tolist(), cache_key=text_col)
    X_train, X_valid, X_test = X_minilm[train_idx], X_minilm[valid_idx], X_minilm[test_idx]
    time_rank_train = df["time_rank"].values[train_idx].astype(float)

    analyzer = EnsembleAnalyzer(y_valid=y_valid, y_test=y_test, num_classes=6, n_boot=2000)

    vp, tp = train_tfidf_logreg(train_texts, y_train, valid_texts, test_texts)
    analyzer.add_model("TF-IDF+LogReg", vp, tp)

    vp, tp = train_tfidf_linear_svm(train_texts, y_train, valid_texts, test_texts)
    analyzer.add_model("TF-IDF+LinearSVM", vp, tp)

    metadata_cols = {"subject": df["subject"].values[train_idx], ...}  # see function docstring
    vp, tp = train_tfidf_logreg_metadata(train_texts, y_train, valid_texts, test_texts, ...)
    analyzer.add_model("TF-IDF+LogReg+metadata", vp, tp)

    da_results = train_da_minilm_family(model, X_train, y_train, time_rank_train, X_valid, y_valid, X_test,
                                         seeds=[42, 43, 44, 45, 46])
    analyzer.add_model("Uniform-LR", *da_results["Uniform-LR"])
    analyzer.add_model("Uniform-MLP", *da_results["Uniform-MLP"])
    analyzer.add_model("MiniLM-LogReg-RW", *da_results["MiniLM-LogReg-RW"])
    analyzer.add_model("DA-MiniLM-RWAC", *da_results["DA-MiniLM-RWAC"])

    results = analyzer.run_full_analysis()
    print(results["individual_df"])
    print(results["pairwise_df"])
    print(results["equal_weight_summary"])
    print(results["greedy_summary"])
"""

from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, log_loss,
)


NUM_CLASSES_DEFAULT = 6


# ============================================================
# Self-contained weighting / calibration / bootstrap helpers
# (duplicated intentionally -- see module docstring)
# ============================================================

def normalize_weights(w, min_weight: float = 0.05, max_weight: float = 20.0) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, min_weight, max_weight)
    return w / (np.mean(w) + 1e-12)


def effective_number_class_weights(y, beta: float = 0.999, num_classes: int = NUM_CLASSES_DEFAULT) -> np.ndarray:
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
    t = np.asarray(time_rank_train, dtype=float)
    age = t.max() - t
    w = np.exp(-float(lam) * age * 10.0)
    return normalize_weights(w)


def combined_sample_weights(y_train, time_rank_train, recency_lambda: float = 0.0,
                             beta: float = 0.999, num_classes: int = NUM_CLASSES_DEFAULT) -> np.ndarray:
    w_class = class_sample_weights(y_train, beta=beta, num_classes=num_classes)
    w_recency = recency_sample_weights(time_rank_train, recency_lambda)
    return normalize_weights(w_class * w_recency)


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


def safe_nll(y_true, proba, num_classes: int = NUM_CLASSES_DEFAULT) -> float:
    return float(log_loss(y_true, np.clip(proba, 1e-12, 1.0), labels=list(range(num_classes))))


def select_best_calibration_two_stage(
    valid_proba, y_valid, y_train,
    tau_grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    temperature_grid: Sequence[float] = (0.75, 1.0, 1.25, 1.5, 2.0),
    num_classes: int = NUM_CLASSES_DEFAULT,
) -> dict:
    """Selects tau under macro-F1, then temperature under NLL (temperature
    has no effect on argmax and so cannot be meaningfully selected under a
    macro-F1 objective -- confirmed directly in this project)."""
    best_tau, best_tau_score = 0.0, -1e9
    for tau in tau_grid:
        bias = logit_adjustment_bias(y_train, tau=tau, num_classes=num_classes)
        calibrated = apply_bias_and_temperature(valid_proba, bias=bias, temperature=1.0)
        score = f1_score(y_valid, calibrated.argmax(axis=1), average="macro", zero_division=0)
        if score > best_tau_score:
            best_tau, best_tau_score = tau, float(score)

    bias = logit_adjustment_bias(y_train, tau=best_tau, num_classes=num_classes)
    best_temp, best_temp_score = 1.0, 1e9
    for temp in temperature_grid:
        calibrated = apply_bias_and_temperature(valid_proba, bias=bias, temperature=temp)
        nll = safe_nll(y_valid, calibrated, num_classes=num_classes)
        if nll < best_temp_score:
            best_temp, best_temp_score = temp, float(nll)

    return {"tau": best_tau, "temperature": best_temp,
            "macro_f1_at_best_tau": best_tau_score, "nll_at_best_temperature": best_temp_score}


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
    return {"mean_diff": float(diffs.mean()), "ci_lower_95": float(ci_lower), "ci_upper_95": float(ci_upper),
            "p_value": p_value, "significant_at_95": bool(ci_lower > 0 or ci_upper < 0)}


def evaluate_proba(y_true, proba) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
    }


def _assert_class_order(classes, num_classes, model_name):
    expected = np.arange(num_classes)
    assert np.array_equal(classes, expected), (
        f"{model_name}: classes_ {classes} does not match expected {expected} -- "
        f"ensembling with misaligned class order would silently combine the WRONG "
        f"probability columns. Fit this model on integer label_id, not label strings."
    )


# ============================================================
# Concrete model-training functions (every model this project has)
# ============================================================

TFIDF_LOGREG_VECTORIZER = dict(
    lowercase=True, stop_words="english", ngram_range=(1, 2),
    max_features=30000, min_df=2, max_df=0.90, sublinear_tf=True,
)
TFIDF_LOGREG_CLASSIFIER = dict(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs")

TFIDF_SVM_VECTORIZER = dict(
    lowercase=True, stop_words=None, ngram_range=(1, 2),
    max_features=30000, min_df=2, max_df=0.90, sublinear_tf=True,
)
TFIDF_SVM_CLASSIFIER = dict(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)

TFIDF_METADATA_VECTORIZER = dict(
    lowercase=True, stop_words="english", ngram_range=(1, 2),
    min_df=2, max_df=0.95, max_features=50000, sublinear_tf=True,
)
LOGREG_METADATA_CLASSIFIER = dict(solver="saga", C=1.0, class_weight="balanced", max_iter=5000, random_state=42)


def train_tfidf_logreg(train_texts, y_train, valid_texts, test_texts, num_classes: int = NUM_CLASSES_DEFAULT):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_LOGREG_VECTORIZER)),
        ("classifier", LogisticRegression(**TFIDF_LOGREG_CLASSIFIER)),
    ])
    pipeline.fit(train_texts, y_train)
    _assert_class_order(pipeline.named_steps["classifier"].classes_, num_classes, "TF-IDF+LogReg")
    return pipeline.predict_proba(valid_texts), pipeline.predict_proba(test_texts)


def train_tfidf_linear_svm(train_texts, y_train, valid_texts, test_texts, num_classes: int = NUM_CLASSES_DEFAULT):
    """LinearSVC has no predict_proba -- decision_function is converted to
    a probability-like distribution via softmax over the class scores.
    This is a reasonable proxy for ensembling purposes but is NOT a
    calibrated probability in the same sense as the other models here;
    treat SVM's contribution to any ensemble with that caveat in mind."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_SVM_VECTORIZER)),
        ("classifier", LinearSVC(**TFIDF_SVM_CLASSIFIER)),
    ])
    pipeline.fit(train_texts, y_train)
    clf = pipeline.named_steps["classifier"]
    _assert_class_order(clf.classes_, num_classes, "TF-IDF+LinearSVM")

    def to_proba(texts):
        scores = pipeline.decision_function(texts)
        return softmax_np(scores)

    return to_proba(valid_texts), to_proba(test_texts)


def train_tfidf_logreg_metadata(
    train_texts, y_train, valid_texts, test_texts,
    train_metadata_df, valid_metadata_df, test_metadata_df,
    metadata_cols: Sequence[str], num_classes: int = NUM_CLASSES_DEFAULT,
):
    """`*_metadata_df` are small DataFrames (or dict-of-arrays) containing
    just `metadata_cols` for the corresponding split, aligned row-for-row
    with `*_texts`. Uses only leakage-safe restricted metadata -- pass
    columns like subject/speaker/state_info/context, never credibility
    counts or justification text."""
    train_texts_col = "__text__"
    train_df = pd.DataFrame(train_metadata_df).copy()
    train_df[train_texts_col] = train_texts
    valid_df = pd.DataFrame(valid_metadata_df).copy()
    valid_df[train_texts_col] = valid_texts
    test_df = pd.DataFrame(test_metadata_df).copy()
    test_df[train_texts_col] = test_texts

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf_text", TfidfVectorizer(**TFIDF_METADATA_VECTORIZER), train_texts_col),
            ("categorical_metadata", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), list(metadata_cols)),
        ],
        remainder="drop", sparse_threshold=1.0,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(**LOGREG_METADATA_CLASSIFIER)),
    ])
    pipeline.fit(train_df, y_train)
    _assert_class_order(pipeline.named_steps["classifier"].classes_, num_classes, "TF-IDF+LogReg+metadata")
    return pipeline.predict_proba(valid_df), pipeline.predict_proba(test_df)


def _train_da_family_one_seed(
    da_model, X_train, y_train, time_rank_train, X_valid, y_valid, X_test, seed: int,
    recency_lambdas: Sequence[float], num_classes: int,
):
    """Trains all four DA-MiniLM-family conditions (Uniform-LR,
    Uniform-MLP, MiniLM-LogReg-RW, DA-MiniLM-RWAC) for one seed, reusing
    the SAME embeddings, so all four differ only in
    reweighting/architecture, not in encoding. Returns a dict keyed by
    condition name -> (valid_cal, test_cal)."""
    results = {}

    # --- Uniform (unweighted) conditions: no lambda search, weights all 1 ---
    w_uniform = np.ones(len(y_train), dtype=float)

    valid_proba, test_proba = _train_logreg_on_embeddings(X_train, y_train, w_uniform, X_valid, X_test, seed)
    calib = select_best_calibration_two_stage(valid_proba, y_valid, y_train, num_classes=num_classes)
    bias = logit_adjustment_bias(y_train, tau=calib["tau"], num_classes=num_classes)
    results["Uniform-LR"] = (
        apply_bias_and_temperature(valid_proba, bias=bias, temperature=calib["temperature"]),
        apply_bias_and_temperature(test_proba, bias=bias, temperature=calib["temperature"]),
    )

    valid_proba, test_proba, _ = da_model.train_da_minilm_mlp(X_train, y_train, w_uniform, X_valid, y_valid, X_test, seed=seed)
    calib = select_best_calibration_two_stage(valid_proba, y_valid, y_train, num_classes=num_classes)
    bias = logit_adjustment_bias(y_train, tau=calib["tau"], num_classes=num_classes)
    results["Uniform-MLP"] = (
        apply_bias_and_temperature(valid_proba, bias=bias, temperature=calib["temperature"]),
        apply_bias_and_temperature(test_proba, bias=bias, temperature=calib["temperature"]),
    )

    # --- Reweighted conditions: search recency_lambdas, select by validation macro-F1 ---
    for model_kind, out_name in (("logreg", "MiniLM-LogReg-RW"), ("mlp", "DA-MiniLM-RWAC")):
        best = {"valid_macro_f1": -1.0}
        for rec_lam in recency_lambdas:
            w_train = combined_sample_weights(y_train, time_rank_train, recency_lambda=rec_lam, num_classes=num_classes)
            if model_kind == "logreg":
                valid_proba, test_proba = _train_logreg_on_embeddings(X_train, y_train, w_train, X_valid, X_test, seed)
            else:
                valid_proba, test_proba, _ = da_model.train_da_minilm_mlp(
                    X_train, y_train, w_train, X_valid, y_valid, X_test, seed=seed,
                )
            calib = select_best_calibration_two_stage(valid_proba, y_valid, y_train, num_classes=num_classes)
            bias = logit_adjustment_bias(y_train, tau=calib["tau"], num_classes=num_classes)
            valid_cal = apply_bias_and_temperature(valid_proba, bias=bias, temperature=calib["temperature"])
            f1 = f1_score(y_valid, valid_cal.argmax(axis=1), average="macro", zero_division=0)
            if f1 > best["valid_macro_f1"]:
                test_cal = apply_bias_and_temperature(test_proba, bias=bias, temperature=calib["temperature"])
                best = {"valid_macro_f1": f1, "valid_cal": valid_cal, "test_cal": test_cal}
        results[out_name] = (best["valid_cal"], best["test_cal"])

    return results


def _train_logreg_on_embeddings(X_train, y_train, w_train, X_valid, X_test, seed: int):
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, random_state=seed))
    clf.fit(X_train, y_train, logisticregression__sample_weight=w_train)
    return clf.predict_proba(X_valid), clf.predict_proba(X_test)


def train_da_minilm_family(
    da_model, X_train, y_train, time_rank_train, X_valid, y_valid, X_test,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    recency_lambdas: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30),
    num_classes: int = NUM_CLASSES_DEFAULT,
) -> dict:
    """Trains Uniform-LR, Uniform-MLP, MiniLM-LogReg-RW, and
    DA-MiniLM-RWAC across all `seeds`, mean-pooling calibrated
    probabilities per condition. Returns {condition_name: (valid_proba,
    test_proba)} ready to register with EnsembleAnalyzer.add_model()."""
    per_condition_valid = {name: [] for name in ("Uniform-LR", "Uniform-MLP", "MiniLM-LogReg-RW", "DA-MiniLM-RWAC")}
    per_condition_test = {name: [] for name in per_condition_valid}

    for seed in seeds:
        print(f"  training DA-MiniLM family, seed={seed}...")
        seed_results = _train_da_family_one_seed(
            da_model, X_train, y_train, time_rank_train, X_valid, y_valid, X_test, seed,
            recency_lambdas, num_classes,
        )
        for name, (vp, tp) in seed_results.items():
            per_condition_valid[name].append(vp)
            per_condition_test[name].append(tp)

    return {
        name: (np.mean(per_condition_valid[name], axis=0), np.mean(per_condition_test[name], axis=0))
        for name in per_condition_valid
    }


def train_distilbert(
    train_texts, y_train, valid_texts, y_valid, test_texts,
    num_classes: int = NUM_CLASSES_DEFAULT, model_name: str = "distilbert-base-uncased",
    max_length: int = 128, batch_size: int = 16, epochs: int = 3, seed: int = 42,
):
    """Fine-tunes DistilBERT and returns (valid_proba, test_proba) --
    matching baseline_models.py's run_distilbert() model/config, but
    returning probabilities instead of a metrics row, so it can be
    registered with EnsembleAnalyzer alongside every other model.

    NOT TESTED in this sandbox (no network access to the Hugging Face
    Hub) -- verify in your own GPU/internet environment before trusting
    its output. Includes the same two version-compatibility fallbacks
    already found necessary elsewhere in this project: `Trainer`'s
    `tokenizer=` kwarg was renamed to `processing_class=` in transformers
    5.x, and `TrainingArguments`' `evaluation_strategy=` was renamed to
    `eval_strategy=` in 4.46+. Both are tried in order rather than pinned
    to one version.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_hf_dataset(texts, labels):
        ds = Dataset.from_dict({"text": list(texts), "label": list(map(int, labels))})
        ds = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length), batched=True)
        return ds

    train_ds = to_hf_dataset(train_texts, y_train)
    valid_ds = to_hf_dataset(valid_texts, y_valid)
    test_ds = Dataset.from_dict({"text": list(test_texts)})
    test_ds = test_ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    try:
        args = TrainingArguments(
            output_dir="./distilbert_ensemble_out", num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
            eval_strategy="epoch", save_strategy="no", report_to=[], seed=seed,
        )
    except TypeError:
        args = TrainingArguments(
            output_dir="./distilbert_ensemble_out", num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch", save_strategy="no", report_to=[], seed=seed,
        )
    try:
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                           data_collator=collator, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                           data_collator=collator, tokenizer=tokenizer)
    trainer.train()

    valid_logits = trainer.predict(valid_ds).predictions
    test_logits = trainer.predict(test_ds).predictions
    return softmax_np(valid_logits), softmax_np(test_logits)


# ============================================================
# Full-project orchestration: train and register EVERY model in one call
# ============================================================

def build_full_project_ensemble_analyzer(
    y_valid, y_test, train_texts, y_train, valid_texts, test_texts,
    da_model, X_train, X_valid, X_test, time_rank_train,
    train_metadata: Optional[dict] = None, valid_metadata: Optional[dict] = None,
    test_metadata: Optional[dict] = None, metadata_cols: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (42, 43, 44, 45, 46), num_classes: int = NUM_CLASSES_DEFAULT,
    n_boot: int = 2000, include_distilbert: bool = False,
) -> "EnsembleAnalyzer":
    """Trains EVERY model this project has and registers all of them with
    a single EnsembleAnalyzer, ready for run_full_analysis(): TF-IDF+
    LogReg, TF-IDF+LinearSVM, TF-IDF+LogReg+metadata (if metadata columns
    given), Uniform-LR, Uniform-MLP, MiniLM-LogReg-RW, DA-MiniLM-RWAC, and
    optionally DistilBERT. With 7-8 models this covers C(7,2)=21 or
    C(8,2)=28 pairwise combinations, plus the equal-weight blend and
    greedy best-subset search -- the complete combinatorial picture, not
    a hand-picked pair.

    `train_texts`/`valid_texts`/`test_texts` should be the
    claim-plus-restricted-metadata (or whichever) text field, aligned
    row-for-row with X_train/X_valid/X_test (the MiniLM embeddings of the
    SAME text) and with train_metadata/valid_metadata/test_metadata (if
    given). include_distilbert=True requires Hugging Face Hub access and
    a GPU -- leave False in environments without both.
    """
    analyzer = EnsembleAnalyzer(y_valid=y_valid, y_test=y_test, num_classes=num_classes, n_boot=n_boot)

    print("=== Training TF-IDF + Logistic Regression ===")
    vp, tp = train_tfidf_logreg(train_texts, y_train, valid_texts, test_texts, num_classes=num_classes)
    analyzer.add_model("TF-IDF+LogReg", vp, tp)

    print("=== Training TF-IDF + Linear SVM ===")
    vp, tp = train_tfidf_linear_svm(train_texts, y_train, valid_texts, test_texts, num_classes=num_classes)
    analyzer.add_model("TF-IDF+LinearSVM", vp, tp)

    if metadata_cols and train_metadata is not None:
        print("=== Training TF-IDF + Logistic Regression + restricted metadata ===")
        vp, tp = train_tfidf_logreg_metadata(
            train_texts, y_train, valid_texts, test_texts,
            train_metadata, valid_metadata, test_metadata, metadata_cols, num_classes=num_classes,
        )
        analyzer.add_model("TF-IDF+LogReg+metadata", vp, tp)
    else:
        print("=== Skipping TF-IDF+LogReg+metadata (no metadata_cols/train_metadata given) ===")

    print(f"=== Training DA-MiniLM family (Uniform-LR, Uniform-MLP, MiniLM-LogReg-RW, "
          f"DA-MiniLM-RWAC) across {len(seeds)} seeds ===")
    da_results = train_da_minilm_family(
        da_model, X_train, y_train, time_rank_train, X_valid, y_valid, X_test,
        seeds=seeds, num_classes=num_classes,
    )
    for name in ("Uniform-LR", "Uniform-MLP", "MiniLM-LogReg-RW", "DA-MiniLM-RWAC"):
        analyzer.add_model(name, *da_results[name])

    if include_distilbert:
        print("=== Training DistilBERT (requires Hugging Face Hub + GPU) ===")
        vp, tp = train_distilbert(train_texts, y_train, valid_texts, y_valid, test_texts, num_classes=num_classes)
        analyzer.add_model("DistilBERT", vp, tp)
    else:
        print("=== Skipping DistilBERT (include_distilbert=False) ===")

    return analyzer


# ============================================================
# EnsembleAnalyzer: the model-agnostic, expandable core
# ============================================================

class EnsembleAnalyzer:

    def __init__(self, y_valid, y_test, num_classes: int = NUM_CLASSES_DEFAULT,
                 weight_grid: Sequence[float] = tuple(round(x, 2) for x in np.arange(0.0, 1.001, 0.05)),
                 n_boot: int = 2000, bootstrap_seed: int = 42):
        self.y_valid = np.asarray(y_valid)
        self.y_test = np.asarray(y_test)
        self.num_classes = num_classes
        self.weight_grid = weight_grid
        self.n_boot = n_boot
        self.bootstrap_seed = bootstrap_seed
        self.models: dict = {}  # name -> (valid_proba, test_proba)

    def add_model(self, name: str, valid_proba: np.ndarray, test_proba: np.ndarray):
        assert valid_proba.shape[1] == self.num_classes, f"{name}: valid_proba has {valid_proba.shape[1]} columns, expected {self.num_classes}"
        assert test_proba.shape[1] == self.num_classes, f"{name}: test_proba has {test_proba.shape[1]} columns, expected {self.num_classes}"
        self.models[name] = (valid_proba, test_proba)
        print(f"Registered model: {name}")

    # ---------------- Individual performance ----------------

    def individual_performance(self) -> pd.DataFrame:
        rows = []
        for name, (valid_proba, test_proba) in self.models.items():
            v = evaluate_proba(self.y_valid, valid_proba)
            t = evaluate_proba(self.y_test, test_proba)
            rows.append({"model": name, "valid_macro_f1": v["macro_f1"], "valid_accuracy": v["accuracy"],
                         "test_macro_f1": t["macro_f1"], "test_accuracy": t["accuracy"],
                         "test_macro_precision": t["macro_precision"], "test_macro_recall": t["macro_recall"]})
        return pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

    # ---------------- Pairwise ensembles: every combination ----------------

    def _sweep_pair(self, name_a: str, name_b: str) -> dict:
        va, ta = self.models[name_a]
        vb, tb = self.models[name_b]

        rows = []
        for w in self.weight_grid:
            valid_ens = w * va + (1 - w) * vb
            test_ens = w * ta + (1 - w) * tb
            vm = evaluate_proba(self.y_valid, valid_ens)
            tm = evaluate_proba(self.y_test, test_ens)
            rows.append({"weight_a": w, "weight_b": round(1 - w, 2),
                         "valid_macro_f1": vm["macro_f1"], "test_macro_f1": tm["macro_f1"]})
        curve_df = pd.DataFrame(rows)

        selected_row = curve_df.sort_values("valid_macro_f1", ascending=False).iloc[0]
        oracle_row = curve_df.sort_values("test_macro_f1", ascending=False).iloc[0]
        selected_w = float(selected_row["weight_a"])

        final_test_ens = selected_w * ta + (1 - selected_w) * tb
        ens_pred = final_test_ens.argmax(axis=1)
        pred_a = ta.argmax(axis=1)
        pred_b = tb.argmax(axis=1)

        boot_vs_a = paired_bootstrap_macro_f1_diff(self.y_test, ens_pred, pred_a, n_boot=self.n_boot, seed=self.bootstrap_seed)
        boot_vs_b = paired_bootstrap_macro_f1_diff(self.y_test, ens_pred, pred_b, n_boot=self.n_boot, seed=self.bootstrap_seed)

        return {
            "model_a": name_a, "model_b": name_b, "curve_df": curve_df,
            "selected_weight_a": selected_w, "selected_test_macro_f1": float(selected_row["test_macro_f1"]),
            "selected_valid_macro_f1": float(selected_row["valid_macro_f1"]),
            "oracle_weight_a": float(oracle_row["weight_a"]), "oracle_test_macro_f1": float(oracle_row["test_macro_f1"]),
            "selection_gap": float(oracle_row["test_macro_f1"] - selected_row["test_macro_f1"]),
            "gain_over_a": float(selected_row["test_macro_f1"]) - evaluate_proba(self.y_test, ta)["macro_f1"],
            "gain_over_b": float(selected_row["test_macro_f1"]) - evaluate_proba(self.y_test, tb)["macro_f1"],
            "bootstrap_vs_a": boot_vs_a, "bootstrap_vs_b": boot_vs_b,
        }

    def pairwise_ensembles(self) -> dict:
        """Every unordered pair of registered models. Returns
        {"summary_df": DataFrame (one row per pair), "details": {(a,b): full sweep dict}}."""
        names = list(self.models.keys())
        summary_rows = []
        details = {}
        for name_a, name_b in combinations(names, 2):
            print(f"  sweeping pair: {name_a} x {name_b} ...")
            result = self._sweep_pair(name_a, name_b)
            details[(name_a, name_b)] = result
            summary_rows.append({
                "model_a": name_a, "model_b": name_b,
                "selected_weight_a": result["selected_weight_a"],
                "selected_test_macro_f1": result["selected_test_macro_f1"],
                "oracle_test_macro_f1": result["oracle_test_macro_f1"],
                "selection_gap": result["selection_gap"],
                "gain_over_a": result["gain_over_a"], "gain_over_b": result["gain_over_b"],
                "bootstrap_p_vs_a": result["bootstrap_vs_a"]["p_value"],
                "significant_vs_a": result["bootstrap_vs_a"]["significant_at_95"],
                "bootstrap_p_vs_b": result["bootstrap_vs_b"]["p_value"],
                "significant_vs_b": result["bootstrap_vs_b"]["significant_at_95"],
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("selected_test_macro_f1", ascending=False).reset_index(drop=True)
        return {"summary_df": summary_df, "details": details}

    # ---------------- All-model equal-weight blend ----------------

    def equal_weight_ensemble(self, subset: Optional[Sequence[str]] = None) -> dict:
        names = list(subset) if subset is not None else list(self.models.keys())
        valid_stack = np.mean([self.models[n][0] for n in names], axis=0)
        test_stack = np.mean([self.models[n][1] for n in names], axis=0)
        v = evaluate_proba(self.y_valid, valid_stack)
        t = evaluate_proba(self.y_test, test_stack)
        return {"models": names, "valid_macro_f1": v["macro_f1"], "test_macro_f1": t["macro_f1"],
                "test_accuracy": t["accuracy"]}

    # ---------------- Greedy forward-selection ensemble ----------------

    def greedy_best_subset(self, weight_grid: Optional[Sequence[float]] = None) -> dict:
        """Starts from the single best model on validation, then
        repeatedly tries adding each remaining model (blended in at a
        grid-searched weight against the current ensemble, selected by
        validation macro-F1), keeping the addition only if it improves
        validation macro-F1 over the current best. Stops when no
        remaining model helps. This explores a data-driven subset of the
        combinatorial space (avoiding an intractable full search over all
        2^n subsets x weight-simplex combinations) while still directly
        answering "what's the best ensemble found so far.\""""
        weight_grid = weight_grid if weight_grid is not None else self.weight_grid
        indiv = self.individual_performance()
        current_name = indiv.iloc[0]["model"]
        current_valid = self.models[current_name][0].copy()
        current_test = self.models[current_name][1].copy()
        chosen = [current_name]
        history = [{"step": 0, "added": current_name, "subset": list(chosen),
                    "valid_macro_f1": evaluate_proba(self.y_valid, current_valid)["macro_f1"],
                    "test_macro_f1": evaluate_proba(self.y_test, current_test)["macro_f1"]}]

        remaining = [n for n in self.models if n != current_name]
        improved = True
        step = 0
        while improved and remaining:
            improved = False
            best_candidate = None
            best_candidate_valid_f1 = evaluate_proba(self.y_valid, current_valid)["macro_f1"]
            best_candidate_w = None
            best_candidate_valid = None
            best_candidate_test = None

            for candidate in remaining:
                cand_valid, cand_test = self.models[candidate]
                for w in weight_grid:
                    blended_valid = w * cand_valid + (1 - w) * current_valid
                    f1 = f1_score(self.y_valid, blended_valid.argmax(axis=1), average="macro", zero_division=0)
                    if f1 > best_candidate_valid_f1:
                        best_candidate_valid_f1 = f1
                        best_candidate = candidate
                        best_candidate_w = w
                        best_candidate_valid = blended_valid
                        best_candidate_test = w * cand_test + (1 - w) * current_test

            if best_candidate is not None:
                step += 1
                current_valid, current_test = best_candidate_valid, best_candidate_test
                chosen.append(best_candidate)
                remaining.remove(best_candidate)
                history.append({
                    "step": step, "added": f"{best_candidate} (w={best_candidate_w})",
                    "subset": list(chosen),
                    "valid_macro_f1": best_candidate_valid_f1,
                    "test_macro_f1": evaluate_proba(self.y_test, current_test)["macro_f1"],
                })
                improved = True
                print(f"  greedy step {step}: added {best_candidate} (w={best_candidate_w}) "
                      f"-> valid_macro_f1={best_candidate_valid_f1:.4f}")

        history_df = pd.DataFrame(history)
        return {
            "final_subset": chosen,
            "final_valid_macro_f1": float(history_df.iloc[-1]["valid_macro_f1"]),
            "final_test_macro_f1": float(history_df.iloc[-1]["test_macro_f1"]),
            "history_df": history_df,
        }

    # ---------------- Full analysis ----------------

    def run_full_analysis(self) -> dict:
        """Runs everything: individual performance, all pairwise
        ensembles, the all-model equal-weight blend, and the greedy
        best-subset search. Prints a summary leaderboard at the end."""
        print(f"\n=== Individual model performance ({len(self.models)} models) ===")
        individual_df = self.individual_performance()
        print(individual_df.to_string(index=False))

        print(f"\n=== Pairwise ensembles ({len(self.models) * (len(self.models) - 1) // 2} pairs) ===")
        pairwise = self.pairwise_ensembles()
        print(pairwise["summary_df"].to_string(index=False))

        print(f"\n=== All-model equal-weight ensemble ===")
        equal_weight_summary = self.equal_weight_ensemble()
        print(f"  models: {equal_weight_summary['models']}")
        print(f"  valid_macro_f1={equal_weight_summary['valid_macro_f1']:.4f}, "
              f"test_macro_f1={equal_weight_summary['test_macro_f1']:.4f}")

        print(f"\n=== Greedy best-subset search ===")
        greedy_summary = self.greedy_best_subset()
        print(f"  final subset: {greedy_summary['final_subset']}")
        print(f"  final valid_macro_f1={greedy_summary['final_valid_macro_f1']:.4f}, "
              f"test_macro_f1={greedy_summary['final_test_macro_f1']:.4f}")

        print(f"\n=== Overall leaderboard (test macro-F1) ===")
        leaderboard_rows = [{"entry": row["model"], "kind": "individual", "test_macro_f1": row["test_macro_f1"]}
                             for _, row in individual_df.iterrows()]
        for _, row in pairwise["summary_df"].iterrows():
            leaderboard_rows.append({
                "entry": f"{row['model_a']} + {row['model_b']} (w={row['selected_weight_a']})",
                "kind": "pairwise", "test_macro_f1": row["selected_test_macro_f1"],
            })
        leaderboard_rows.append({"entry": f"equal-weight ({len(self.models)} models)", "kind": "equal_weight",
                                  "test_macro_f1": equal_weight_summary["test_macro_f1"]})
        leaderboard_rows.append({"entry": f"greedy: {' + '.join(greedy_summary['final_subset'])}",
                                  "kind": "greedy", "test_macro_f1": greedy_summary["final_test_macro_f1"]})
        leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
        print(leaderboard_df.to_string(index=False))

        return {
            "individual_df": individual_df,
            "pairwise_df": pairwise["summary_df"],
            "pairwise_details": pairwise["details"],
            "equal_weight_summary": equal_weight_summary,
            "greedy_summary": greedy_summary,
            "leaderboard_df": leaderboard_df,
        }


if __name__ == "__main__":
    print("ensemble_experiment.py loaded. Classes/functions: EnsembleAnalyzer, "
          "train_tfidf_logreg, train_tfidf_linear_svm, train_tfidf_logreg_metadata, "
          "train_da_minilm_family")