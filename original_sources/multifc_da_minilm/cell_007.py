# ============================================================
# 6. Metrics, calibration, weighting, and statistical helpers
# ============================================================

def ordinal_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def within_one_accuracy(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def expected_calibration_error(y_true, y_pred, proba, n_bins=10):
    if proba is None:
        return np.nan
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)
    confidence = np.max(proba, axis=1)
    correct = (y_true == y_pred).astype(float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence <= hi) if i == n_bins - 1 else (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def multiclass_brier_score(y_true, proba, n_classes=NUM_CLASSES):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    onehot = np.zeros((len(y_true), n_classes), dtype=float)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def compute_metrics(y_true, y_pred, proba=None):
    row = {
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
        row["mean_confidence"] = float(np.max(proba, axis=1).mean())
        row["ece"] = expected_calibration_error(y_true, y_pred, proba)
        row["brier"] = multiclass_brier_score(y_true, proba, n_classes=NUM_CLASSES)
        row["nll"] = float(log_loss(y_true, np.clip(proba, 1e-12, 1.0), labels=list(range(NUM_CLASSES))))
    return row


def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 1e-6, None)
    return w / np.mean(w)


def effective_num_class_weights(y, beta=0.999):
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / np.mean(weights[counts > 0])
    return weights[y]


def recency_exp_weights_from_time_rank(time_rank, recency_lambda=0.0):
    t = np.asarray(time_rank, dtype=float)
    if len(t) <= 1 or recency_lambda == 0:
        return np.ones(len(t), dtype=float)
    # Normalize inside the training subset, so the oldest training item is 0 and newest is 1.
    t_norm = (t - np.min(t)) / max(np.max(t) - np.min(t), 1e-12)
    w = np.exp(float(recency_lambda) * t_norm)
    return normalize_weights(w)


def combined_sample_weights(y, time_rank, recency_lambda=0.0, beta=0.999):
    class_w = effective_num_class_weights(y, beta=beta)
    rec_w = recency_exp_weights_from_time_rank(time_rank, recency_lambda=recency_lambda)
    return normalize_weights(class_w * rec_w)


def softmax_np(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def logit_adjustment_bias(y_train, tau=1.0):
    counts = np.bincount(np.asarray(y_train).astype(int), minlength=NUM_CLASSES).astype(float) + 1e-6
    prior = counts / counts.sum()
    # Negative prior correction can reduce majority-class overprediction.
    return -float(tau) * np.log(prior)


def apply_bias_and_temperature(proba, bias=None, temperature=1.0):
    logits = np.log(np.clip(proba, 1e-12, 1.0))
    if bias is not None:
        logits = logits + np.asarray(bias).reshape(1, -1)
    logits = logits / max(float(temperature), 1e-6)
    return softmax_np(logits)


def select_best_calibration(valid_proba, y_valid, y_train, objective="macro_f1"):
    best = None
    for tau in TAU_GRID:
        bias = logit_adjustment_bias(y_train, tau=tau)
        for temp in TEMPERATURE_GRID:
            adj = apply_bias_and_temperature(valid_proba, bias=bias, temperature=temp)
            pred = adj.argmax(axis=1)
            if objective == "macro_f1":
                score = f1_score(y_valid, pred, average="macro", zero_division=0)
            elif objective == "nll":
                score = -log_loss(y_valid, np.clip(adj, 1e-12, 1.0), labels=list(range(NUM_CLASSES)))
            else:
                raise ValueError(objective)
            candidate = {"tau": tau, "temperature": temp, "valid_score": float(score)}
            if best is None or score > best["valid_score"]:
                best = candidate
    return best


def paired_bootstrap_macro_f1_diff(y_true, pred_a, pred_b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fa = f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
        fb = f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
        diffs.append(fa - fb)
    diffs = np.asarray(diffs)
    obs = f1_score(y_true, pred_a, average="macro", zero_division=0) - f1_score(y_true, pred_b, average="macro", zero_division=0)
    # Two-sided bootstrap p-value around zero.
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p = min(float(p), 1.0)
    return {
        "observed_macro_f1_diff": float(obs),
        "bootstrap_mean_diff": float(diffs.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_value_raw": p,
        "significant_95_raw": bool((np.quantile(diffs, 0.025) > 0) or (np.quantile(diffs, 0.975) < 0)),
    }


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[order[i]] = prev
    return np.clip(q, 0.0, 1.0)


def summarize_results(df, group_cols):
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "ordinal_mae", "within_one_accuracy", "ece", "brier", "nll"]
    agg = {}
    for m in metrics:
        if m in df.columns:
            agg[m] = ["mean", "std"]
    out = df.groupby(group_cols).agg(agg)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    if "seed" in df.columns:
        seed_counts = df.groupby(group_cols)["seed"].nunique().reset_index(name="n_seeds")
        out = out.merge(seed_counts, on=group_cols, how="left")
    return out