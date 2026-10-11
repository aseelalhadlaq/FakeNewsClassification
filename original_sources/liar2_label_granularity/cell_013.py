prep = Liar2Preprocessor()
df6 = prep.run_pipeline(LIAR2_TRAIN, LIAR2_VALID, LIAR2_TEST)

# Canonical dedup (keep_first) to reproduce reported cohort sizes, then
# re-establish chronological row order + time_rank on the deduped frame.
try:
    _audit = prep.audit_duplicate_statements(df6)
    df6 = prep.drop_duplicate_statements(df6, policy="keep_first", audit_df=_audit)
    df6 = prep.parse_and_sort_by_date(df6)   # re-sort after dropping rows
    df6 = prep.add_time_rank(df6)            # recompute time_rank over final rows
    print("Applied dedup(keep_first) + chronological re-sort + time_rank.")
except Exception as e:
    print("NOTE: dedup step skipped or unavailable ->", repr(e))

df6 = df6.reset_index(drop=True)   # positions must align with split indices

assert "label_id" in df6.columns and "time_rank" in df6.columns
assert f"model_text__{TEXT_MODE}" in df6.columns, (
    f"Expected column model_text__{TEXT_MODE} not found. "
    f"Available model_text columns: {[c for c in df6.columns if c.startswith('model_text__')]}"
)
print(f"\nFinal LIAR2 rows: {len(df6)}")
print("6-class label_id distribution:")
print(df6["label_id"].value_counts().sort_index()
      .rename(index={i: n for i, n in enumerate(SIXCLASS_ORDER)}))