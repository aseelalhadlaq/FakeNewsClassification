# ============================================================
# 11. Paired bootstrap with Benjamini-Hochberg correction
# ============================================================

bootstrap_rows = []

for text_mode in TEXT_MODES:
    for eval_setup in EVAL_SETUPS:
        for seed in SEEDS:
            key_a = ("DA-MiniLM++", text_mode, eval_setup, seed)
            key_b = ("MiniLM-LogReg-RW", text_mode, eval_setup, seed)
            if key_a not in prediction_store or key_b not in prediction_store:
                continue
            y_test = prediction_store[key_a]["y_test"]
            pred_a = prediction_store[key_a]["pred"]
            pred_b = prediction_store[key_b]["pred"]
            boot = paired_bootstrap_macro_f1_diff(y_test, pred_a, pred_b, n_boot=2000, seed=seed)
            bootstrap_rows.append({
                "comparison": "DA-MiniLM++ minus MiniLM-LogReg-RW",
                "text_mode": text_mode,
                "eval_setup": eval_setup,
                "seed": seed,
                **boot,
            })

bootstrap_df = pd.DataFrame(bootstrap_rows)
if len(bootstrap_df) > 0:
    bootstrap_df["q_value_bh"] = benjamini_hochberg(bootstrap_df["p_value_raw"].values)
    bootstrap_df["significant_05_bh"] = bootstrap_df["q_value_bh"] <= 0.05

bootstrap_path = OUTPUT_DIR / "paired_bootstrap_da_minilm_vs_logreg_with_bh.csv"
bootstrap_df.to_csv(bootstrap_path, index=False)
print("Saved:", bootstrap_path)
display(bootstrap_df)

# Per-condition count of seeds improved/significant after BH.
if len(bootstrap_df) > 0:
    boot_summary = (
        bootstrap_df.assign(improved=lambda d: d["observed_macro_f1_diff"] > 0)
        .groupby(["text_mode", "eval_setup"])
        .agg(
            n_tests=("seed", "count"),
            n_improved=("improved", "sum"),
            mean_diff=("observed_macro_f1_diff", "mean"),
            n_significant_bh=("significant_05_bh", "sum"),
        )
        .reset_index()
    )
    boot_summary_path = OUTPUT_DIR / "paired_bootstrap_condition_summary.csv"
    boot_summary.to_csv(boot_summary_path, index=False)
    print("Saved:", boot_summary_path)
    display(boot_summary)