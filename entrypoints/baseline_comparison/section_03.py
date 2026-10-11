# ============================================================
# SECTION 3: DistilBERT chronological, multi-seed (a previously
# outstanding required experiment -- only in the baseline battery
# above under the OFFICIAL split, not chronological)
# ============================================================

if INCLUDE_TRANSFORMER_BASELINES:
    distilbert_chrono_df, distilbert_chrono_summary = baselines.run_distilbert_multiseed(
        liar2_df, "LIAR2 chronological", seeds=SEEDS, split_mode="chronological",
    )
    distilbert_chrono_df.to_csv(OUTPUT_ROOT / "distilbert_chronological_multiseed.csv", index=False)
    print("DistilBERT chronological summary:", distilbert_chrono_summary)
else:
    print("Skipping DistilBERT-chronological (INCLUDE_TRANSFORMER_BASELINES=False).")

print("=" * 70)
print("SECTION 3 complete.")
print("=" * 70)


