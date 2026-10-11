# ============================================================
# SECTION 10: Rolling-window temporal evaluation
# ============================================================

rolling_results_df = da_runner.run_rolling_window_evaluation(
    liar2_final_df, da_embeddings, text_mode="claim_plus_safe_metadata",
    seeds=SEEDS, model_kind="da_minilm_mlp",
)
rolling_results_df.to_csv(OUTPUT_ROOT / "da_minilm_rolling_window.csv", index=False)

print("=" * 70)
print(f"SECTION 10 complete: {len(rolling_results_df)} rolling-window rows saved.")
print("=" * 70)


