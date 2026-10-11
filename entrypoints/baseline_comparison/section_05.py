# ============================================================
# SECTION 5: Drift diagnostics (D01-D05) on the deduplicated
# chronological cohort
# ============================================================

#from drift_analysis import DriftAnalysis

da = DriftAnalysis()

INCLUDE_EMBEDDING_SHIFT = INCLUDE_TRANSFORMER_BASELINES  # D04 needs Hugging Face Hub too

drift_summary_df = da.run_all_drift_experiments(
    drift_train_df, drift_test_df, include_embedding_shift=INCLUDE_EMBEDDING_SHIFT,
)
drift_summary_df.to_csv(OUTPUT_ROOT / "drift_summary.csv", index=False)

drift_comparison_df = da.compare_to_reported(drift_summary_df)
drift_comparison_df.to_csv(OUTPUT_ROOT / "drift_comparison_vs_reported.csv", index=False)

print("=" * 70)
print("SECTION 5 complete: drift diagnostics saved.")
print("=" * 70)


