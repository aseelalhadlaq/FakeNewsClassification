prep = MultiFCPreprocessor(output_dir=OUTPUT_DIR)


def _resolve(path, fname):
    """Fall back to a recursive Kaggle-input search if the literal path is missing."""
    if Path(path).exists():
        return path
    for base in ["/kaggle/input", "/mnt/data", "."]:
        b = Path(base)
        if b.exists():
            hits = list(b.rglob(fname))
            if hits:
                return str(hits[0])
    raise FileNotFoundError(f"Could not locate {fname}. Edit the Section 0 paths.")


_train = _resolve(MULTIFC_TRAIN, "train.tsv")
_dev   = _resolve(MULTIFC_DEV,   "dev.tsv")
_test  = _resolve(MULTIFC_TEST,  "test.tsv") if Path(MULTIFC_TEST).name else None
print("Resolved paths:\n  train:", _train, "\n  dev  :", _dev, "\n  test :", _test)

df3 = prep.run_pipeline(_train, _dev, _test)

# save the label-harmonisation audit required by the manuscript
prep.label_harmonization_report(
    save_path=f"{OUTPUT_DIR}/multifc_label_harmonization_report.csv")

# Canonical dedup (keep_first) then re-establish chronological order + time_rank.
try:
    _audit = prep.audit_duplicate_statements(df3)
    df3 = prep.drop_duplicate_statements(df3, policy="keep_first", audit_df=_audit)
    df3 = prep.parse_and_sort_by_date(df3)
    df3 = prep.add_time_rank(df3)
    print("Applied dedup(keep_first) + chronological re-sort + time_rank.")
except Exception as e:
    print("NOTE: dedup step skipped or unavailable ->", repr(e))

df3 = df3.reset_index(drop=True)

assert "label_id" in df3.columns and "time_rank" in df3.columns
assert f"model_text__{TEXT_MODE}" in df3.columns, (
    f"Expected column model_text__{TEXT_MODE} not found. Available: "
    f"{[c for c in df3.columns if c.startswith('model_text__')]}")
print(f"\nFinal MultiFC rows: {len(df3)}")
print("3-class label_id distribution:")
print(df3["label_id"].value_counts().sort_index()
      .rename(index={i: n for i, n in enumerate(THREECLASS_ORDER)}))
