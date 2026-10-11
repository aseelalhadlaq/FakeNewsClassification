# ============================================================
# SECTION 4: LIAR2 deduplicated + chronologically split cohort
# (used by BOTH the drift diagnostics and the DA-MiniLM experiments,
# matching the notebook this project's central claim is built on)
# ============================================================

audit = liar2_prep.audit_duplicate_statements(liar2_df)
liar2_dedup_df = liar2_prep.drop_duplicate_statements(liar2_df, policy="keep_first", audit_df=audit)
liar2_dedup_df = liar2_dedup_df.reset_index(drop=True)

chrono_train_idx, chrono_valid_idx, chrono_test_idx = liar2_prep.chronological_split(liar2_dedup_df)
drift_train_df = liar2_dedup_df.iloc[chrono_train_idx].copy()
drift_test_df = liar2_dedup_df.iloc[chrono_test_idx].copy()

print(f"Deduplicated chronological cohort: "
      f"train={len(chrono_train_idx)}, valid={len(chrono_valid_idx)}, test={len(chrono_test_idx)}")
print("=" * 70)
print("SECTION 4 complete.")
print("=" * 70)


