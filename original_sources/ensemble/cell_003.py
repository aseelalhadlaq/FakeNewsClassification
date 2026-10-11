#from liar2_preprocessor import Liar2Preprocessor
#from da_minilm_model import DAMiniLMModel
#from ensemble_experiment import build_full_project_ensemble_analyzer
train_path ="/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/train.csv"
valid_path = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/valid.csv"
test_path = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/test.csv"
prep = Liar2Preprocessor()
df = prep.run_pipeline(train_path, valid_path, test_path).reset_index(drop=True)
df = prep.add_time_rank(df)

train_idx, valid_idx, test_idx = prep.chronological_split(df)
y_all = df["label_id"].values.astype(int)
y_train, y_valid, y_test = y_all[train_idx], y_all[valid_idx], y_all[test_idx]

text_col = "model_text__claim_plus_safe_metadata"
texts_all = df[text_col].values
train_texts, valid_texts, test_texts = texts_all[train_idx], texts_all[valid_idx], texts_all[test_idx]

meta_cols = ["subject", "speaker", "state_info", "context"]
train_meta = {c: df[c].values[train_idx] for c in meta_cols}
valid_meta = {c: df[c].values[valid_idx] for c in meta_cols}
test_meta = {c: df[c].values[test_idx] for c in meta_cols}

model = DAMiniLMModel(output_dir="./ensemble_results")
X_minilm = model.encode_texts_minilm(df[text_col].tolist(), cache_key=text_col)
X_train, X_valid, X_test = X_minilm[train_idx], X_minilm[valid_idx], X_minilm[test_idx]
time_rank_train = df["time_rank"].values[train_idx].astype(float)

analyzer = build_full_project_ensemble_analyzer(
    y_valid=y_valid, y_test=y_test,
    train_texts=train_texts, y_train=y_train, valid_texts=valid_texts, test_texts=test_texts,
    da_model=model, X_train=X_train, X_valid=X_valid, X_test=X_test, time_rank_train=time_rank_train,
    train_metadata=train_meta, valid_metadata=valid_meta, test_metadata=test_meta, metadata_cols=meta_cols,
    seeds=[42, 43, 44, 45, 46], n_boot=2000,
    include_distilbert=False,   # set True only with Hugging Face Hub access + GPU
)

results = analyzer.run_full_analysis()