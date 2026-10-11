# ============================================================
# 4. Build model text inputs and audit final sample sizes
# ============================================================

TEXT_MODE_DESCRIPTIONS = {
    "claim_only": "Only the claim text.",
    "claim_plus_safe_metadata": "Claim plus category, speaker, source_site, and entities.",
    "claim_plus_safe_metadata_and_title": "Adds title; not recommended because title can contain verdict leakage.",
    "claim_plus_evidence": "Adds evidence; not recommended for no-evidence fake-news detection.",
}


def build_model_text(row, mode="claim_plus_safe_metadata"):
    claim = clean_text(row.get("claim", ""))
    if mode == "claim_only":
        return claim

    parts = [claim]
    if mode in ["claim_plus_safe_metadata", "claim_plus_safe_metadata_and_title", "claim_plus_evidence"]:
        if clean_text(row.get("category", "")):
            parts.append("category: " + clean_text(row.get("category", "")))
        if clean_text(row.get("speaker", "")):
            parts.append("speaker: " + clean_text(row.get("speaker", "")))
        if clean_text(row.get("source_site", "")):
            parts.append("source: " + clean_text(row.get("source_site", "")))
        if clean_text(row.get("entities", "")):
            parts.append("entities: " + clean_text(row.get("entities", "")))

    if mode == "claim_plus_safe_metadata_and_title":
        if clean_text(row.get("title", "")):
            parts.append("title: " + clean_text(row.get("title", "")))

    if mode == "claim_plus_evidence":
        if clean_text(row.get("evidence_text", "")):
            parts.append("evidence: " + clean_text(row.get("evidence_text", "")))

    return " [SEP] ".join([p for p in parts if p])

# Create all text modes once.
for mode in TEXT_MODES:
    for df in [train_std, dev_std, test_std]:
        df[f"text__{mode}"] = df.apply(lambda r: build_model_text(r, mode=mode), axis=1)

# Use claim text availability as the primary inclusion criterion.
train_lab = train_std[
    train_std["has_label"] & train_std["label_3class"].isin(LABEL_ORDER) & (train_std["claim"].str.len() > 0)
].copy().reset_index(drop=True)

dev_lab = dev_std[
    dev_std["has_label"] & dev_std["label_3class"].isin(LABEL_ORDER) & (dev_std["claim"].str.len() > 0)
].copy().reset_index(drop=True)

test_unlabelled = test_std[(test_std["claim"].str.len() > 0)].copy().reset_index(drop=True)

# Preserve split identity for official evaluation.
labelled_all = pd.concat([train_lab, dev_lab], ignore_index=True).reset_index(drop=True)

# Build a stable chronological order.
# If valid dates are available for at least half the labelled data, sort by date.
# Otherwise, use official split order + row order as a temporal proxy.
labelled_all["split_order"] = labelled_all["split"].map({"train": 0, "dev": 1, "test": 2}).fillna(9).astype(int)
valid_date_ratio = float(labelled_all["date"].notna().mean())
USE_DATE_ORDER = valid_date_ratio >= 0.50

if USE_DATE_ORDER:
    labelled_all = labelled_all.sort_values(["date", "split_order", "row_id"], na_position="last").reset_index(drop=True)
    print(f"Chronological order uses parsed dates. Valid date ratio={valid_date_ratio:.3f}")
else:
    labelled_all = labelled_all.sort_values(["split_order", "row_id"]).reset_index(drop=True)
    print(f"Chronological order uses split/order proxy. Valid date ratio={valid_date_ratio:.3f}")

labelled_all["chrono_index"] = np.arange(len(labelled_all))
labelled_all["time_rank"] = labelled_all["chrono_index"] / max(len(labelled_all) - 1, 1)

# Audit trail to prevent N mismatches.
audit_rows = []
def audit(stage, df):
    audit_rows.append({
        "stage": stage,
        "n_rows": len(df),
        "n_has_label": int(df["has_label"].sum()) if "has_label" in df.columns else np.nan,
        "n_mapped_label": int(df["label_3class"].isin(LABEL_ORDER).sum()) if "label_3class" in df.columns else np.nan,
        "n_nonempty_claim": int((df["claim"].astype(str).str.len() > 0).sum()) if "claim" in df.columns else np.nan,
    })

audit("raw_train", train_raw)
audit("raw_dev", dev_raw)
audit("raw_test", test_raw)
audit("standardized_train", train_std)
audit("standardized_dev", dev_std)
audit("usable_train_labelled", train_lab)
audit("usable_dev_labelled", dev_lab)
audit("usable_labelled_all", labelled_all)
audit("usable_test_unlabelled", test_unlabelled)

audit_df = pd.DataFrame(audit_rows)
audit_path = OUTPUT_DIR / "sample_size_audit.csv"
audit_df.to_csv(audit_path, index=False)
print("Saved sample-size audit:", audit_path)
display(audit_df)

print("\nFinal label distributions:")
print("train_lab:")
print(train_lab["label_3class"].value_counts())
print("\ndev_lab:")
print(dev_lab["label_3class"].value_counts())
print("\nlabelled_all:")
print(labelled_all["label_3class"].value_counts())

display(labelled_all[["split", "claim_id", "date", "label_3class", "time_rank", "claim"]].head())