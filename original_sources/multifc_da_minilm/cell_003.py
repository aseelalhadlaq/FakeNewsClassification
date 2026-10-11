# ============================================================
# 2. MultiFC file discovery and TSV loading
# ============================================================

def find_file(filename):
    candidates = [
        Path(filename),
        Path("/kaggle/input") / filename,
        Path("/kaggle/input/multifc") / filename,
        Path("/kaggle/input/multifc-dataset") / filename,
        Path("/kaggle/input/multifc-drift-aware") / filename,
        Path("/kaggle/working") / filename,
        Path("/mnt/data") / filename,
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(list(kaggle_input.rglob(filename)))
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename}. Please update the path manually.")

TRAIN_PATH = find_file("train.tsv")
DEV_PATH   = find_file("dev.tsv")
TEST_PATH  = find_file("test.tsv")

print("TRAIN_PATH:", TRAIN_PATH)
print("DEV_PATH  :", DEV_PATH)
print("TEST_PATH :", TEST_PATH)

LABELLED_COLUMNS = [
    "claim_id", "claim", "raw_label", "url", "evidence_text",
    "category", "speaker", "checker", "tags", "title",
    "published_date", "claim_date", "entities"
]

UNLABELLED_COLUMNS = [
    "claim_id", "claim", "url", "evidence_text",
    "category", "speaker", "checker", "tags", "title",
    "published_date", "claim_date", "entities"
]


def read_multifc_tsv(path, split_name):
    raw = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
        quoting=3,
        engine="python",
    )
    if raw.shape[1] == 13:
        raw.columns = LABELLED_COLUMNS
        raw["has_label"] = True
    elif raw.shape[1] == 12:
        raw.columns = UNLABELLED_COLUMNS
        raw["raw_label"] = np.nan
        raw["has_label"] = False
    else:
        raise ValueError(f"{path} has {raw.shape[1]} columns. Expected 13 labelled or 12 unlabelled columns.")

    raw["split"] = split_name
    raw["row_id"] = np.arange(len(raw))
    raw["global_row_order"] = raw["row_id"]
    raw["source_site"] = (
        raw["claim_id"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False).fillna("unknown")
    )
    return raw

train_raw = read_multifc_tsv(TRAIN_PATH, "train")
dev_raw   = read_multifc_tsv(DEV_PATH, "dev")
test_raw  = read_multifc_tsv(TEST_PATH, "test")

print("Raw shapes:")
print("train:", train_raw.shape, "has_label:", bool(train_raw["has_label"].iloc[0]))
print("dev  :", dev_raw.shape, "has_label:", bool(dev_raw["has_label"].iloc[0]))
print("test :", test_raw.shape, "has_label:", bool(test_raw["has_label"].iloc[0]))

display(train_raw.head(3))
display(dev_raw.head(3))
display(test_raw.head(3))