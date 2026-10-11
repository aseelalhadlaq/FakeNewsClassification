# ============================================================
# SECTION 9: Shuffle-control validation (label permutation null test)
# -- uses the fast LogReg head, 300 shuffles, on the same condition as
# the ablation above
# ============================================================

#from shuffle_control import ShuffleControlValidator

X_shuf = da_embeddings[ABLATION_TEXT_MODE]
y_shuf = liar2_final_df["label_id"].values.astype(int)

X_train_s, y_train_s = X_shuf[ablation_train_idx], y_shuf[ablation_train_idx]
X_valid_s, y_valid_s = X_shuf[ablation_valid_idx], y_shuf[ablation_valid_idx]
X_test_s, y_test_s = X_shuf[ablation_test_idx], y_shuf[ablation_test_idx]


def _shuffle_train_predict_fn(X_tr, y_tr, X_va, y_va, X_te, seed):
    w = np.ones(len(y_tr))
    return da_model.train_logreg_baseline(X_tr, y_tr, w, X_va, y_va, X_te, seed=seed)


shuffle_validator = ShuffleControlValidator(_shuffle_train_predict_fn, n_shuffles=300, seed=42)
shuffle_result = shuffle_validator.run(X_train_s, y_train_s, X_valid_s, y_valid_s, X_test_s, y_test_s)
shuffle_validator.print_summary(shuffle_result)

pd.DataFrame([shuffle_result["summary"]]).to_csv(OUTPUT_ROOT / "shuffle_control_summary.csv", index=False)
np.save(OUTPUT_ROOT / "shuffle_control_null_scores.npy", shuffle_result["null_scores"])

print("=" * 70)
print("SECTION 9 complete.")
print("=" * 70)


