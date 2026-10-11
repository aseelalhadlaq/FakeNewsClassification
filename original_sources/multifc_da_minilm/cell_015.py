# ============================================================
# 14. Official unlabelled test.tsv predictions
# ============================================================

# This cell trains DA-MiniLM++ on the labelled train+dev data with an inner validation split,
# then predicts labels for the official unlabelled test.tsv.
# It is for generating submission/prediction files only, not for reporting test performance.

if RUN_OFFICIAL_TEST_PREDICTIONS and len(test_unlabelled) > 0 and RUN_DA_MINILM_PLUS:
    FINAL_TEXT_MODE = "claim_plus_safe_metadata"
    X_labelled = all_embeddings[FINAL_TEXT_MODE]
    X_official_test = test_embeddings[FINAL_TEXT_MODE]

    # Inner chronological split from labelled_all for lambda/calibration selection.
    ordered = labelled_all.sort_values("time_rank").index.to_numpy()
    cut = int(0.85 * len(ordered))
    final_train_idx = ordered[:cut]
    final_valid_idx = ordered[cut:]

    final_prediction_frames = []
    for seed in SEEDS:
        X_train = X_labelled[final_train_idx]
        y_train = y_all[final_train_idx]
        t_train = time_rank_all[final_train_idx]
        X_valid = X_labelled[final_valid_idx]
        y_valid = y_all[final_valid_idx]

        best = None
        payload = None
        for rec_lam in RECENCY_LAMBDAS:
            w_train = combined_sample_weights(y_train, t_train, recency_lambda=rec_lam)
            valid_proba, test_proba, raw_valid_f1 = train_da_minilm_mlp(
                X_train, y_train, w_train, X_valid, y_valid, X_official_test, seed=seed
            )
            calib = select_best_calibration(valid_proba, y_valid, y_train, objective="macro_f1")
            valid_cal = apply_bias_and_temperature(
                valid_proba,
                bias=logit_adjustment_bias(y_train, tau=calib["tau"]),
                temperature=calib["temperature"],
            )
            valid_pred = valid_cal.argmax(axis=1)
            valid_macro_f1 = f1_score(y_valid, valid_pred, average="macro", zero_division=0)
            if best is None or valid_macro_f1 > best["valid_macro_f1"]:
                best = {"seed": seed, "recency_lambda": rec_lam, "valid_macro_f1": float(valid_macro_f1), **calib}
                payload = test_proba

        test_cal = apply_bias_and_temperature(
            payload,
            bias=logit_adjustment_bias(y_train, tau=best["tau"]),
            temperature=best["temperature"],
        )
        pred_id = test_cal.argmax(axis=1)
        pred_label = [id_to_label[int(i)] for i in pred_id]

        pred_df = test_unlabelled[["claim_id", "claim", "source_site", "category", "speaker"]].copy()
        pred_df["seed"] = seed
        pred_df["pred_label"] = pred_label
        pred_df["pred_id"] = pred_id
        for i, lab in id_to_label.items():
            pred_df[f"proba_{lab}"] = test_cal[:, i]
        pred_df["selected_recency_lambda"] = best["recency_lambda"]
        pred_df["selected_tau"] = best["tau"]
        pred_df["selected_temperature"] = best["temperature"]
        pred_df["inner_valid_macro_f1"] = best["valid_macro_f1"]
        final_prediction_frames.append(pred_df)

    all_seed_pred_df = pd.concat(final_prediction_frames, ignore_index=True)
    all_seed_path = OUTPUT_DIR / "official_test_predictions_all_seeds.csv"
    all_seed_pred_df.to_csv(all_seed_path, index=False)
    print("Saved:", all_seed_path)

    # Majority vote / mean probability ensemble across seeds.
    proba_cols = [f"proba_{lab}" for lab in LABEL_ORDER]
    ensemble = (
        all_seed_pred_df.groupby(["claim_id", "claim", "source_site", "category", "speaker"], dropna=False)[proba_cols]
        .mean()
        .reset_index()
    )
    ensemble_proba = ensemble[proba_cols].values
    ensemble["pred_id"] = ensemble_proba.argmax(axis=1)
    ensemble["pred_label"] = [id_to_label[int(i)] for i in ensemble["pred_id"]]
    ensemble_path = OUTPUT_DIR / "official_test_predictions_seed_ensemble.csv"
    ensemble.to_csv(ensemble_path, index=False)
    print("Saved:", ensemble_path)
    display(ensemble.head())
else:
    print("Official test predictions skipped.")