#from liar2_preprocessor import Liar2Preprocessor

p_liar2 = Liar2Preprocessor()
liar2_dataset = p_liar2.run_pipeline(
    "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/train.csv",
    "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/valid.csv",
    "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/test.csv",
    
)