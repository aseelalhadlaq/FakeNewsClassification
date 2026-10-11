def build_runner(num_classes):
    """Fresh model + runner configured for a given class count. A new model is
    required so num_classes is correct for the MLP head and for Brier/NLL; a new
    runner so class-balance weights are recomputed on the collapsed priors."""
    model = DAMiniLMModel(output_dir=f"{OUTPUT_DIR}/K{num_classes}", num_classes=num_classes)
    # reuse the already-computed embedding cache so we never re-encode
    model._embedding_cache = _encoder._embedding_cache
    splitter = TemporalSplitter(train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15)
    runner = DAMiniLMExperimentRunner(
        splitter=splitter, model=model, num_classes=num_classes,
        output_dir=f"{OUTPUT_DIR}/K{num_classes}",
    )
    return model, splitter, runner


all_main_rows = []
all_bootstrap_rows = []
prediction_stores = {}      # scheme -> prediction_store from run_full_battery

for scheme in SCHEMES:
    cfg = COLLAPSE_SCHEMES[scheme]
    K = cfg["num_classes"]
    print("\n" + "=" * 72)
    print(f"SCHEME = {scheme}   (num_classes = {K})")
    print("=" * 72)

    collapser = LabelCollapser(scheme)
    df_c = collapser.transform(df6).reset_index(drop=True)
    print(collapser.describe_collapse(df_c).to_string(index=False))

    os.makedirs(f"{OUTPUT_DIR}/K{K}", exist_ok=True)
    model, splitter, runner = build_runner(K)

    # ---- DA-MiniLM-RWAC + MiniLM-LogReg-RW via the standard battery ----
    main_df, lambda_df, pred_store = runner.run_full_battery(
        df=df_c, embeddings=embeddings, seeds=SEEDS,
        text_modes=TEXT_MODES, split_modes=[SPLIT_MODE],
        model_kinds=("minilm_logreg", "da_minilm_mlp"),
        calibration_mode="two_stage",
    )

    # ---- Uniform-LR (unweighted frozen-embedding LogReg) ----
    y_all = df_c["label_id"].values.astype(int)
    t_all = df_c["time_rank"].values.astype(float)
    for _mode in TEXT_MODES:
        X = embeddings[_mode]
        for seed in SEEDS:
            tr, va, te = splitter.get_split(df_c, SPLIT_MODE, seed=seed)
            row_u, _, pred_u, proba_u = runner.run_single_experiment(
                X, y_all, t_all, tr, va, te,
                text_mode=_mode, split_mode=SPLIT_MODE, seed=seed,
                model_kind="minilm_logreg", use_reweighting=False,
                use_calibration=True, calibration_mode="two_stage",
            )
            main_df = pd.concat([main_df, pd.DataFrame([row_u])], ignore_index=True)
            pred_store[(row_u["model"], _mode, SPLIT_MODE, seed)] = {
                "y_test": y_all[te], "pred": pred_u, "proba": proba_u,
            }

    main_df["scheme"] = scheme
    main_df["num_classes"] = K
    all_main_rows.append(main_df)
    prediction_stores[scheme] = pred_store

    # ---- Paired bootstrap: DA-MiniLM-RWAC minus MiniLM-LogReg-RW ----
    boot_df = runner.run_paired_bootstrap(
        pred_store, text_modes=TEXT_MODES, split_modes=[SPLIT_MODE],
        seeds=SEEDS, model_a="DA-MiniLM-RWAC", model_b="MiniLM-LogReg-RW",
        n_boot=N_BOOT,
    )
    if len(boot_df):
        boot_df["scheme"] = scheme
        boot_df["num_classes"] = K
        all_bootstrap_rows.append(boot_df)

main_results = pd.concat(all_main_rows, ignore_index=True)
bootstrap_results = (pd.concat(all_bootstrap_rows, ignore_index=True)
                     if all_bootstrap_rows else pd.DataFrame())

main_results.to_csv(f"{OUTPUT_DIR}/collapse_main_results.csv", index=False)
if len(bootstrap_results):
    bootstrap_results.to_csv(f"{OUTPUT_DIR}/collapse_bootstrap_results.csv", index=False)

print("\nAll schemes complete.")
print("Main results rows:", len(main_results))