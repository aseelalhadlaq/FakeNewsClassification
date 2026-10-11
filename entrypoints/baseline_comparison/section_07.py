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


