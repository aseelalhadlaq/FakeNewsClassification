# ============================================================
# SECTION 0: Paths -- EDIT THESE to match your Kaggle input structure
# ============================================================

LIAR_TRAIN = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar dataset/liar dataset/train.tsv"
LIAR_VALID = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar dataset/liar dataset/valid.tsv"
LIAR_TEST = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar dataset/liar dataset/test.tsv"

LIAR2_TRAIN = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/train.csv"
LIAR2_VALID = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/valid.csv"
LIAR2_TEST = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/test.csv"

# Verify these three against your actual MultiFC dataset folder listing
# before running -- filenames/casing vary by upload (e.g. train.tsv vs
# train.tsv.txt). List the folder first if unsure:
#     for p in sorted(Path(MULTIFC_FOLDER).rglob("*")): print(p)
MULTIFC_TRAIN = "/kaggle/input/datasets/badalahmmed/multifc/train.tsv"
MULTIFC_DEV = "/kaggle/input/datasets/badalahmmed/multifc/dev.tsv"
MULTIFC_TEST = "/kaggle/input/datasets/badalahmmed/multifc/test.tsv"

OUTPUT_ROOT = Path("/kaggle/working/full_run_results")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
TEXT_MODES = ["claim_only", "claim_plus_safe_metadata"]

print("=" * 70)
print("SECTION 0 complete: paths configured, output root:", OUTPUT_ROOT)
print("=" * 70)


