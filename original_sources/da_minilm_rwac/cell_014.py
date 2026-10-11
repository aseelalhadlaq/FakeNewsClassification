# LIAR2 loading + canonical preprocessing

p_liar2 = Liar2Preprocessor()
liar2_dataset = p_liar2.run_pipeline(
    "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/train.csv",
    "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/valid.csv",
    "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/test.csv",
)

assert len(liar2_dataset) == 22923
print(f"Final deduplicated LIAR2 rows: {len(liar2_dataset):,}")
