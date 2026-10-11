# ============================================================
# SECTION 1: Load and preprocess LIAR and LIAR2
# ============================================================

#from liar_preprocessor import LiarPreprocessor
#from liar2_preprocessor import Liar2Preprocessor

liar_prep = LiarPreprocessor()
liar_df = liar_prep.run_pipeline(LIAR_TRAIN, LIAR_VALID, LIAR_TEST)

liar2_prep = Liar2Preprocessor()
liar2_df = liar2_prep.run_pipeline(LIAR2_TRAIN, LIAR2_VALID, LIAR2_TEST)

print("=" * 70)
print(f"SECTION 1 complete: LIAR {len(liar_df)} rows, LIAR2 {len(liar2_df)} rows")
print("=" * 70)


