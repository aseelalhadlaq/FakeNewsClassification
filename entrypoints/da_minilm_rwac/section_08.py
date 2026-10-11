# ============================================================
# SECTION 8: Component ablation -- isolates reweighting's and
# calibration's individual contributions, on the condition carrying
# the central claim (chronological + claim_plus_safe_metadata)
# ============================================================

ABLATION_TEXT_MODE = "claim_plus_safe_metadata"
ABLATION_SPLIT_MODE = "chronological"
ABLATION_SEED = SEEDS[0]

ablation_train_idx, ablation_valid_idx, ablation_test_idx = liar2_prep.chronological_split(liar2_final_df)

ablation_df = da_runner.run_component_ablation(
    da_embeddings[ABLATION_TEXT_MODE],
    liar2_final_df["label_id"].values.astype(int),
    liar2_final_df["time_rank"].values.astype(float),
    ablation_train_idx, ablation_valid_idx, ablation_test_idx,
    text_mode=ABLATION_TEXT_MODE, split_mode=ABLATION_SPLIT_MODE, seed=ABLATION_SEED,
    calibration_mode="two_stage",
)
ablation_df.to_csv(OUTPUT_ROOT / "da_minilm_component_ablation.csv", index=False)

print("Component ablation:")
print(ablation_df[["model", "macro_f1", "use_reweighting", "use_calibration"]])
print("=" * 70)
print("SECTION 8 complete.")
print("=" * 70)


