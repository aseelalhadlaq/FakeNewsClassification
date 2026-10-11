# ============================================================
# 3. Clean text/date and harmonize MultiFC labels
# ============================================================

LABEL_ORDER = ["false", "mixed", "true"]
label_to_id = {label: i for i, label in enumerate(LABEL_ORDER)}
id_to_label = {i: label for label, i in label_to_id.items()}
NUM_CLASSES = len(LABEL_ORDER)


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).replace("\n", " ").replace("\r", " ")
    x = re.sub(r"\s+", " ", x).strip()
    if x.lower() == "none":
        return ""
    return x


def normalize_label_string(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = x.replace("_", "-")
    x = re.sub(r"\s+", " ", x)
    return x

TRUE_LABELS = set(normalize_label_string(x) for x in [
    "true", "truth!", "verdict: true", "determination: true",
    "factscan score: true", "correct", "accurate", "verified", "fact",
    "conclusion: accurate", "mostly true", "mostly-true", "mostly_true",
    "mostly truth!", "mostly-correct", "determination: mostly true",
    "promise kept", "in-the-green", "true messages",
    "authorship confirmed!", "confirmed authorship!",
    "correct attribution", "correct attribution!",
    "truth! & outdated!", "truth! & unproven!",
    "previously truth! now resolved!"
])

MIXED_LABELS = set(normalize_label_string(x) for x in [
    "half-true", "half true", "mixture", "in-between",
    "partly true", "partially true", "needs context", "not the whole story",
    "compromise", "determination: barely true",
    "truth! & fiction!", "truth! & misleading!", "truth! & disputed!",
    "disputed!", "unproven", "unproven!", "unverified",
    "determination: misleading", "factscan score: misleading",
    "misleading", "misleading!", "misleading recommendations",
    "2 pinnochios", "a little baloney", "some baloney",
    "spins the facts", "distorts the facts", "cherry picks",
    "exaggerated", "exaggerates", "understated",
    "half flip", "no flip", "conclusion: unclear", "determination: a stretch"
])

FALSE_LABELS = set(normalize_label_string(x) for x in [
    "false", "mostly false", "mostly-false", "mostly_false",
    "pants on fire!", "pants-fire", "fiction!", "fiction", "mostly fiction!",
    "verdict: false", "determination: false", "factscan score: false",
    "conclusion: false", "rating: false", "we rate this claim false",
    "incorrect", "incorrect attribution!", "inaccurate attribution!",
    "fake", "fake news", "facebook scams", "scam", "scam!",
    "bogus warning", "virus!", "4 pinnochios", "3 pinnochios",
    "a lot of baloney", "full flop", "promise broken",
    "in-the-red", "outdated", "outdated!",
    "misattributed", "miscaptioned", "unsupported",
    "unsubstantiated messages", "verdict: unsubstantiated", "no evidence"
])

DROP_LABELS = set(normalize_label_string(x) for x in [
    "none", "not yet rated", "in the works", "stalled",
    "investigation pending!", "opinion!", "commentary!",
    "unobservable", "grass roots movement!", "legend",
    "fiction! & satire!", "statirical reports",
    "determination: huckster propaganda"
])


def map_multifc_label(raw_label):
    x = normalize_label_string(raw_label)
    if x in TRUE_LABELS:
        return "true"
    if x in MIXED_LABELS:
        return "mixed"
    if x in FALSE_LABELS:
        return "false"
    if x in DROP_LABELS:
        return np.nan
    # Numeric labels are dropped unless their source-specific meaning is manually verified.
    if re.fullmatch(r"\d+", x):
        return np.nan
    return np.nan


def parse_multifc_date(row):
    d1 = pd.to_datetime(row.get("published_date", ""), errors="coerce", utc=True)
    if pd.notna(d1):
        return d1
    d2 = pd.to_datetime(row.get("claim_date", ""), errors="coerce", utc=True)
    if pd.notna(d2):
        return d2
    return pd.NaT


def standardize_multifc(df):
    out = df.copy()
    for col in ["claim", "evidence_text", "category", "speaker", "checker", "tags", "title", "entities", "source_site"]:
        if col in out.columns:
            out[col] = out[col].apply(clean_text)
    out["date"] = out.apply(parse_multifc_date, axis=1)
    out["year"] = out["date"].dt.year
    out["label_3class"] = out["raw_label"].apply(map_multifc_label)
    out["label_id"] = out["label_3class"].map(label_to_id)
    return out

train_std = standardize_multifc(train_raw)
dev_std   = standardize_multifc(dev_raw)
test_std  = standardize_multifc(test_raw)

label_report = (
    pd.concat([train_std, dev_std], ignore_index=True)
    .groupby(["raw_label", "label_3class"], dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
label_report_path = OUTPUT_DIR / "multifc_label_harmonization_report.csv"
label_report.to_csv(label_report_path, index=False)
print("Saved label harmonization report:", label_report_path)

for name, df in [("train", train_std), ("dev", dev_std), ("test", test_std)]:
    if df["has_label"].any():
        print(name, df["label_3class"].value_counts(dropna=False).to_dict())

display(label_report.head(30))