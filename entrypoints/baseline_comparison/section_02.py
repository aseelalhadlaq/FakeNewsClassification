# ============================================================
# SECTION 2: Classical + transformer baselines (LIAR, LIAR2, merge,
# chronological, cross-dataset)
# ============================================================

#from baseline_models import BaselineModels

baselines = BaselineModels(liar_preprocessor=liar_prep, liar2_preprocessor=liar2_prep)

# Set to True only if this notebook has internet + GPU enabled.
INCLUDE_TRANSFORMER_BASELINES = True

baseline_results_df = baselines.run_all_experiments(
    liar_df, liar2_df, include_transformer_baselines=INCLUDE_TRANSFORMER_BASELINES,
)
baseline_results_df.to_csv(OUTPUT_ROOT / "baseline_results.csv", index=False)

print("=" * 70)
print(f"SECTION 2 complete: {len(baseline_results_df)} baseline rows saved.")
print("=" * 70)


