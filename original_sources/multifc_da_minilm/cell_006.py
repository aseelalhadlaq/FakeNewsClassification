# ============================================================
# 5. Split functions: official, chronological, random, rolling
# ============================================================

TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.70, 0.15, 0.15


def split_train_indices_for_official_eval(dataframe, seed=42, validation_mode="chronological"):
    """For official train→dev evaluation, keep dev held out.
    Use only the official training split to create inner train/validation.
    """
    train_idx_all = dataframe.index[dataframe["split"] == "train"].to_numpy()
    dev_idx = dataframe.index[dataframe["split"] == "dev"].to_numpy()
    if len(train_idx_all) == 0 or len(dev_idx) == 0:
        raise ValueError("Official train/dev split not available after filtering.")

    if validation_mode == "chronological":
        train_idx_all = dataframe.loc[train_idx_all].sort_values("time_rank").index.to_numpy()
        cut = max(1, int(round(0.85 * len(train_idx_all))))
        inner_train_idx = train_idx_all[:cut]
        inner_valid_idx = train_idx_all[cut:]
    else:
        y_train_all = dataframe.loc[train_idx_all, "label_id"].astype(int).values
        inner_train_idx, inner_valid_idx = train_test_split(
            train_idx_all,
            test_size=0.15,
            random_state=seed,
            stratify=y_train_all if len(np.unique(y_train_all)) == NUM_CLASSES else None,
        )
    return inner_train_idx, inner_valid_idx, dev_idx


def make_chronological_split_indices(dataframe):
    ordered = dataframe.sort_values("time_rank").index.to_numpy()
    n = len(ordered)
    n_train = int(TRAIN_RATIO * n)
    n_valid = int(VALID_RATIO * n)
    train_idx = ordered[:n_train]
    valid_idx = ordered[n_train:n_train + n_valid]
    test_idx  = ordered[n_train + n_valid:]
    return train_idx, valid_idx, test_idx


def make_random_stratified_split_indices(dataframe, seed=42):
    idx = dataframe.index.to_numpy()
    y = dataframe["label_id"].astype(int).values
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(1 - TRAIN_RATIO),
        random_state=seed,
        stratify=y if len(np.unique(y)) == NUM_CLASSES else None,
    )
    temp_y = dataframe.loc[temp_idx, "label_id"].astype(int).values
    valid_fraction_of_temp = VALID_RATIO / (VALID_RATIO + TEST_RATIO)
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - valid_fraction_of_temp),
        random_state=seed,
        stratify=temp_y if len(np.unique(temp_y)) == NUM_CLASSES else None,
    )
    return train_idx, valid_idx, test_idx


def make_rolling_windows(dataframe, min_train_frac=0.50, valid_frac=0.15, test_frac=0.15, step_frac=0.10):
    ordered = dataframe.sort_values("time_rank").index.to_numpy()
    n = len(ordered)
    windows = []
    start_train_end = int(min_train_frac * n)
    valid_n = max(1, int(valid_frac * n))
    test_n = max(1, int(test_frac * n))
    step_n = max(1, int(step_frac * n))

    window_id = 0
    train_end = start_train_end
    while train_end + valid_n + test_n <= n:
        valid_end = train_end + valid_n
        test_end = valid_end + test_n
        windows.append({
            "window_id": window_id,
            "train_idx": ordered[:train_end],
            "valid_idx": ordered[train_end:valid_end],
            "test_idx": ordered[valid_end:test_end],
            "train_end_frac": train_end / n,
            "valid_end_frac": valid_end / n,
            "test_end_frac": test_end / n,
        })
        window_id += 1
        train_end += step_n
    return windows

# Store split summaries.
split_rows = []
for setup in EVAL_SETUPS:
    if setup == "official_train_to_dev":
        tr, va, te = split_train_indices_for_official_eval(labelled_all, seed=42, validation_mode="chronological")
    elif setup == "internal_chronological":
        tr, va, te = make_chronological_split_indices(labelled_all)
    elif setup == "internal_random_stratified":
        tr, va, te = make_random_stratified_split_indices(labelled_all, seed=42)
    for split_name, ids in [("train", tr), ("valid", va), ("test", te)]:
        counts = labelled_all.loc[ids, "label_3class"].value_counts().to_dict()
        split_rows.append({"eval_setup": setup, "split": split_name, "n": len(ids), **{f"n_{k}": counts.get(k, 0) for k in LABEL_ORDER}})

split_summary_df = pd.DataFrame(split_rows)
split_summary_path = OUTPUT_DIR / "split_summary.csv"
split_summary_df.to_csv(split_summary_path, index=False)
print("Saved split summary:", split_summary_path)
display(split_summary_df)

rolling_windows = make_rolling_windows(labelled_all)
print("Number of rolling windows:", len(rolling_windows))