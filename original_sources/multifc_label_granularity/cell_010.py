# ============================================================
# 1c. TemporalSplitter  (dataset-agnostic; identical to LIAR2 project)
# ============================================================
from sklearn.model_selection import train_test_split


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TemporalSplitter:
    def __init__(self, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15):
        s = train_ratio + valid_ratio + test_ratio
        assert abs(s - 1.0) < 1e-6, f"ratios must sum to 1, got {s}"
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def _chronological(self, df):
        n = len(df)
        tr_end = int(n * self.train_ratio)
        va_end = int(n * (self.train_ratio + self.valid_ratio))
        return (np.arange(0, tr_end), np.arange(tr_end, va_end), np.arange(va_end, n))

    def _random_stratified(self, df, seed=42):
        idx = np.arange(len(df))
        y = df["label_id"].values
        tr, tmp = train_test_split(idx, train_size=self.train_ratio,
                                   random_state=seed, stratify=y)
        rel = self.valid_ratio / (self.valid_ratio + self.test_ratio)
        va, te = train_test_split(tmp, train_size=rel, random_state=seed,
                                  stratify=y[tmp])
        return np.sort(tr), np.sort(va), np.sort(te)

    def get_split(self, df, split_mode, seed=42):
        if split_mode == "chronological":
            return self._chronological(df)
        if split_mode == "random_stratified":
            return self._random_stratified(df, seed=seed)
        raise ValueError(f"Unknown split_mode: {split_mode}")
