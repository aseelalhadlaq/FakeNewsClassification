# ============================================================
# 1e. DAMiniLMExperimentRunner  (dataset-agnostic)
# ============================================================
MODEL_DISPLAY_NAMES = {
    ("da_minilm_mlp", True):  "DA-MiniLM-RWAC",
    ("minilm_logreg", True):  "MiniLM-LogReg-RW",
    ("minilm_logreg", False): "Uniform-LR",
    ("da_minilm_mlp", False): "DA-MiniLM-Unweighted",
}


class DAMiniLMExperimentRunner:
    def __init__(self, splitter, model, num_classes, output_dir):
        self.splitter = splitter
        self.model = model
        self.num_classes = int(num_classes)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- one run ----------
    def run_single_experiment(self, X, y, time_rank, train_idx, valid_idx, test_idx,
                              text_mode, split_mode, seed, model_kind="da_minilm_mlp",
                              use_reweighting=True, use_calibration=True,
                              calibration_mode="two_stage"):
        m = self.model
        Xtr, Xva, Xte = X[train_idx], X[valid_idx], X[test_idx]
        ytr, yva, yte = y[train_idx], y[valid_idx], y[test_idx]
        ttr = time_rank[train_idx]
        model_name = MODEL_DISPLAY_NAMES[(model_kind, use_reweighting)]

        # recency-lambda sweep (only meaningful when reweighting; else single pass)
        lam_grid = RECENCY_LAMBDAS if use_reweighting else [0.0]
        candidates, stored = [], {}
        for lam in lam_grid:
            if use_reweighting:
                w = m.combined_sample_weights(ytr, ttr, recency_lambda=lam)
            else:
                w = np.ones(len(ytr), dtype=np.float32)  # unweighted -> Uniform-LR

            if model_kind == "da_minilm_mlp":
                va_p, te_p, raw_f1 = m.train_da_minilm_mlp(Xtr, ytr, w, Xva, yva, Xte,
                                                           seed=seed)
            elif model_kind == "minilm_logreg":
                va_p, te_p = m.train_logreg(Xtr, ytr, w, Xva, Xte, seed=seed)
                raw_f1 = f1_score(yva, va_p.argmax(1), average="macro", zero_division=0)
            else:
                raise ValueError(f"Unknown model_kind: {model_kind}")

            if use_calibration:
                cal = m.select_best_calibration(va_p, yva, ytr, objective="macro_f1",
                                                calibration_mode=calibration_mode)
            else:
                cal = {"tau": 0.0, "temperature": 1.0}
            bias = m.logit_adjustment_bias(ytr, tau=cal["tau"])
            va_cal = m.apply_bias_and_temperature(va_p, bias=bias,
                                                  temperature=cal["temperature"])
            va_f1 = f1_score(yva, va_cal.argmax(1), average="macro", zero_division=0)

            candidates.append({"recency_lambda": lam, "valid_macro_f1": va_f1,
                               "raw_valid_macro_f1": raw_f1,
                               "tau": cal["tau"], "temperature": cal["temperature"]})
            stored[lam] = (va_p, te_p, cal)

        lambda_df = pd.DataFrame(candidates).sort_values("valid_macro_f1",
                                                         ascending=False)
        best_lam = float(lambda_df.iloc[0]["recency_lambda"])
        va_p, te_p, cal = stored[best_lam]
        bias = m.logit_adjustment_bias(ytr, tau=cal["tau"])
        te_cal = m.apply_bias_and_temperature(te_p, bias=bias,
                                              temperature=cal["temperature"])
        pred = te_cal.argmax(1)

        row = m.evaluate(
            yte, pred, proba=te_cal,
            model=model_name, split_mode=split_mode, text_mode=text_mode, seed=seed,
            recency_lambda=best_lam, tau=cal["tau"], temperature=cal["temperature"],
            n_train=len(train_idx), n_valid=len(valid_idx), n_test=len(test_idx),
            selected_valid_macro_f1=float(lambda_df.iloc[0]["valid_macro_f1"]),
        )
        return row, lambda_df, pred, te_cal

    # ---------- full battery ----------
    def run_full_battery(self, df, embeddings, seeds, text_modes, split_modes,
                         model_kinds=("minilm_logreg", "da_minilm_mlp"),
                         calibration_mode="two_stage"):
        y_all = df["label_id"].values.astype(int)
        t_all = df["time_rank"].values.astype(float)
        main_rows, lambda_rows = [], []
        pred_store = {}
        for text_mode in text_modes:
            X = embeddings[text_mode]
            for split_mode in split_modes:
                for seed in seeds:
                    tr, va, te = self.splitter.get_split(df, split_mode, seed=seed)
                    for kind in model_kinds:
                        row, lam_df, pred, proba = self.run_single_experiment(
                            X, y_all, t_all, tr, va, te,
                            text_mode=text_mode, split_mode=split_mode, seed=seed,
                            model_kind=kind, use_reweighting=True,
                            use_calibration=True, calibration_mode=calibration_mode,
                        )
                        main_rows.append(row)
                        for _, r in lam_df.iterrows():
                            lambda_rows.append({"model": row["model"],
                                                "text_mode": text_mode,
                                                "split_mode": split_mode, "seed": seed,
                                                **r.to_dict()})
                        pred_store[(row["model"], text_mode, split_mode, seed)] = {
                            "y_test": y_all[te], "pred": pred, "proba": proba}
                        print(f"  [{row['model']:>16s}] text={text_mode} "
                              f"split={split_mode} seed={seed}  "
                              f"macroF1={row['macro_f1']:.4f}  "
                              f"lambda={row['recency_lambda']}")
        return (pd.DataFrame(main_rows), pd.DataFrame(lambda_rows), pred_store)

    # ---------- paired bootstrap ----------
    def run_paired_bootstrap(self, prediction_store, text_modes, split_modes, seeds,
                             model_a="DA-MiniLM-RWAC", model_b="MiniLM-LogReg-RW",
                             n_boot=2000):
        rows = []
        for text_mode in text_modes:
            for split_mode in split_modes:
                for seed in seeds:
                    ka = (model_a, text_mode, split_mode, seed)
                    kb = (model_b, text_mode, split_mode, seed)
                    if ka not in prediction_store or kb not in prediction_store:
                        continue
                    ra, rb = prediction_store[ka], prediction_store[kb]
                    y_true = np.asarray(ra["y_test"])
                    pa, pb = np.asarray(ra["pred"]), np.asarray(rb["pred"])
                    rng = np.random.RandomState(seed)
                    n = len(y_true)
                    diffs = np.empty(n_boot)
                    for b in range(n_boot):
                        idx = rng.randint(0, n, size=n)
                        diffs[b] = (
                            f1_score(y_true[idx], pa[idx], average="macro",
                                     zero_division=0)
                            - f1_score(y_true[idx], pb[idx], average="macro",
                                       zero_division=0))
                    lo, hi = np.percentile(diffs, [2.5, 97.5])
                    p = min(2.0 * min(float(np.mean(diffs <= 0)),
                                      float(np.mean(diffs >= 0))), 1.0)
                    rows.append({"text_mode": text_mode, "split_mode": split_mode,
                                 "seed": seed, "model_a": model_a, "model_b": model_b,
                                 "mean_diff": float(diffs.mean()),
                                 "ci_lower_95": float(lo), "ci_upper_95": float(hi),
                                 "p_value": p,
                                 "significant_at_95": bool(lo > 0 or hi < 0)})
        return pd.DataFrame(rows)


print("TemporalSplitter, DAMiniLMModel, DAMiniLMExperimentRunner defined.")
