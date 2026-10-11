"""
temporal_splitter.py

Class 1 of 3: splitting logic only, from da-minilm-main-model-v1-0.ipynb's
cell 4 ("Split functions: random, chronological, and rolling windows").

This is a standalone, dataset-agnostic version of the same splitting
functions Liar2Preprocessor already exposes (chronological_split,
random_stratified_split, make_rolling_windows) -- pulled out on its own
here because the model class and the experiment-runner class in this
same three-class set shouldn't need to know about Liar2Preprocessor
specifically. Any DataFrame with a `label_id` integer column works.

If you're already using Liar2Preprocessor for LIAR2 specifically, its
built-in chronological_split()/random_stratified_split()/
make_rolling_windows() are equivalent to this class's methods (same
logic, verified against the same notebook) -- use whichever is more
convenient for your call site. DAMiniLMExperimentRunner (class 3) accepts
either.

Usage
-----
    from temporal_splitter import TemporalSplitter

    splitter = TemporalSplitter()
    train_idx, valid_idx, test_idx = splitter.chronological_split(df)
    train_idx, valid_idx, test_idx = splitter.random_stratified_split(df, seed=42)
    windows = splitter.make_rolling_windows(df)
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TemporalSplitter:

    def __init__(self, train_ratio: float = 0.70, valid_ratio: float = 0.15, test_ratio: float = 0.15,
                 seed: int = 42):
        assert abs((train_ratio + valid_ratio + test_ratio) - 1.0) < 1e-9, \
            "train_ratio + valid_ratio + test_ratio must sum to 1.0"
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def chronological_split(self, df: pd.DataFrame):
        """Contiguous 70/15/15 split in row order. `df` must already be
        chronologically sorted (e.g. by Liar2Preprocessor's
        parse_and_sort_by_date(), or any equivalent date-ordering step)."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        valid_end = int(n * (self.train_ratio + self.valid_ratio))
        return np.arange(0, train_end), np.arange(train_end, valid_end), np.arange(valid_end, n)

    def random_stratified_split(self, df: pd.DataFrame, seed: Optional[int] = None):
        """70/15/15 stratified-by-label split (label taken from `label_id`)."""
        seed = self.seed if seed is None else seed
        all_idx = np.arange(len(df))
        y = df["label_id"].values

        train_idx, temp_idx = train_test_split(
            all_idx, train_size=self.train_ratio, random_state=seed, stratify=y,
        )
        y_temp = y[temp_idx]
        valid_relative_ratio = self.valid_ratio / (self.valid_ratio + self.test_ratio)
        valid_idx, test_idx = train_test_split(
            temp_idx, train_size=valid_relative_ratio, random_state=seed, stratify=y_temp,
        )
        return np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)

    def get_split(self, df: pd.DataFrame, split_mode: str, seed: Optional[int] = None):
        if split_mode == "chronological":
            return self.chronological_split(df)
        if split_mode == "random_stratified":
            return self.random_stratified_split(df, seed=seed)
        raise ValueError(f"Unknown split_mode: {split_mode}")

    def make_rolling_windows(
        self, df: pd.DataFrame, min_train_frac: float = 0.50, valid_frac: float = 0.10,
        test_frac: float = 0.10, step_frac: float = 0.10,
    ) -> list:
        """Expanding-window temporal evaluation. `df` must already be
        chronologically sorted. Example window: train = earliest 50%,
        valid = next 10%, test = next 10%; the training window then
        expands by step_frac each iteration."""
        n = len(df)
        windows = []
        train_end = int(n * min_train_frac)
        valid_size = int(n * valid_frac)
        test_size = int(n * test_frac)
        step = max(1, int(n * step_frac))

        window_id = 0
        while train_end + valid_size + test_size <= n:
            valid_end = train_end + valid_size
            test_end = valid_end + test_size

            train_idx = np.arange(0, train_end)
            valid_idx = np.arange(train_end, valid_end)
            test_idx = np.arange(valid_end, test_end)

            windows.append({
                "window_id": window_id, "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx,
                "train_end_frac": train_end / n, "valid_end_frac": valid_end / n, "test_end_frac": test_end / n,
                "n_train": len(train_idx), "n_valid": len(valid_idx), "n_test": len(test_idx),
            })
            train_end += step
            window_id += 1

        return windows


if __name__ == "__main__":
    print("temporal_splitter.py loaded. Class: TemporalSplitter")