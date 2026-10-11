# ============================================================
# SECTION 3: DistilBERT chronological, multi-seed
# ============================================================
# Create only the objects required for this chronological DistilBERT experiment.

#from baseline_models import BaselineModels

baselines = BaselineModels(
    liar_preprocessor=liar_prep,
    liar2_preprocessor=liar2_prep,
)
INCLUDE_TRANSFORMER_BASELINES = True

if INCLUDE_TRANSFORMER_BASELINES:
    distilbert_chrono_df, distilbert_chrono_summary = baselines.run_distilbert_multiseed(
        liar2_df, "LIAR2 chronological", seeds=SEEDS, split_mode="chronological",
    )
    distilbert_chrono_df.to_csv(
        OUTPUT_ROOT / "distilbert_chronological_multiseed.csv", index=False
    )
    print("DistilBERT chronological summary:", distilbert_chrono_summary)
else:
    print("Skipping DistilBERT-chronological (INCLUDE_TRANSFORMER_BASELINES=False).")

print("=" * 70)
print(f"SECTION 3 complete on final LIAR2 cohort: {len(liar2_df):,} rows.")
print("=" * 70)


