audit = p_liar2.audit_duplicate_statements(liar2_dataset)
liar2_dataset = p_liar2.drop_duplicate_statements(liar2_dataset, policy="keep_first", audit_df=audit)

train_idx, valid_idx, test_idx = p_liar2.chronological_split(liar2_dataset)
train_df = liar2_dataset.iloc[train_idx].copy()
valid_df = liar2_dataset.iloc[valid_idx].copy()
test_df = liar2_dataset.iloc[test_idx].copy()

print(len(train_df), len(valid_df), len(test_df))  # should print 16046 3438 3439