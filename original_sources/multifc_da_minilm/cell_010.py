# ============================================================
# 9. Experiment runner
# ============================================================

prediction_store = {}
lambda_selection_rows = []


def run_single_minilm_experiment(X, y, time_rank, train_idx, valid_idx, test_idx, text_mode, eval_setup, seed=42, model_kind="da_minilm_mlp"):
    X_train, y_train, t_train = X[train_idx], y[train_idx], time_rank[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    lambda_candidates = []
    candidate_payloads = {}

    for rec_lam in RECENCY_LAMBDAS:
        w_train = combined_sample_weights(y_train, t_train, recency_lambda=rec_lam)

        if model_kind == "da_minilm_mlp":
            valid_proba, test_proba, raw_valid_f1 = train_da_minilm_mlp(
                X_train, y_train, w_train, X_valid, y_valid, X_test, seed=seed
            )
            model_name = "DA-MiniLM++"
        elif model_kind == "minilm_logreg":
            valid_proba, test_proba = train_logreg_baseline(
                X_train, y_train, w_train, X_valid, y_valid, X_test, seed=seed
            )
            raw_valid_f1 = f1_score(y_valid, valid_proba.argmax(axis=1), average="macro", zero_division=0)
            model_name = "MiniLM-LogReg-RW"
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        calib = select_best_calibration(valid_proba, y_valid, y_train, objective="macro_f1")
        valid_cal = apply_bias_and_temperature(
            valid_proba,
            bias=logit_adjustment_bias(y_train, tau=calib["tau"]),
            temperature=calib["temperature"],
        )
        valid_pred = valid_cal.argmax(axis=1)
        valid_macro_f1 = f1_score(y_valid, valid_pred, average="macro", zero_division=0)

        rec = {
            "model": model_name,
            "text_mode": text_mode,
            "eval_setup": eval_setup,
            "seed": seed,
            "recency_lambda": rec_lam,
            "valid_macro_f1": float(valid_macro_f1),
            "raw_valid_macro_f1": float(raw_valid_f1),
            "calibration_tau": calib["tau"],
            "temperature": calib["temperature"],
        }
        lambda_candidates.append(rec)
        candidate_payloads[rec_lam] = (valid_proba, test_proba, calib)

    lambda_df = pd.DataFrame(lambda_candidates).sort_values("valid_macro_f1", ascending=False).reset_index(drop=True)
    best_lam = float(lambda_df.iloc[0]["recency_lambda"])
    valid_proba, test_proba, calib = candidate_payloads[best_lam]

    test_cal = apply_bias_and_temperature(
        test_proba,
        bias=logit_adjustment_bias(y_train, tau=calib["tau"]),
        temperature=calib["temperature"],
    )
    test_pred = test_cal.argmax(axis=1)

    row = {
        "model": model_name,
        "text_mode": text_mode,
        "eval_setup": eval_setup,
        "seed": seed,
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "n_test": len(test_idx),
        "recency_lambda": best_lam,
        "calibration_tau": calib["tau"],
        "temperature": calib["temperature"],
    }
    row.update(compute_metrics(y_test, test_pred, test_cal))

    key = (model_name, text_mode, eval_setup, seed)
    prediction_store[key] = {
        "test_idx": test_idx,
        "y_test": y_test,
        "pred": test_pred,
        "proba": test_cal,
    }

    return row, lambda_df, test_pred, test_cal


def get_eval_indices(eval_setup, seed=42):
    if eval_setup == "official_train_to_dev":
        return split_train_indices_for_official_eval(labelled_all, seed=seed, validation_mode="chronological")
    if eval_setup == "internal_chronological":
        return make_chronological_split_indices(labelled_all)
    if eval_setup == "internal_random_stratified":
        return make_random_stratified_split_indices(labelled_all, seed=seed)
    raise ValueError(eval_setup)