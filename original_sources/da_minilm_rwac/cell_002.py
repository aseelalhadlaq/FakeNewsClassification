"""
liar2_preprocessor.py

One class, LIAR2 only. Every preprocessing step is its own method so you
can call them individually and inspect intermediate output, instead of
only seeing the final combined result.

Two distinct splits are tracked, on purpose, because they answer different
questions:
    - `official_split` : which of train.csv/valid.csv/test.csv a row came
                          from. This is LIAR2's own distributed split
                          (~80/10/10). Useful for reporting results against
                          the split other papers on LIAR2 use.
    - chronological_split() / random_stratified_split() : the project's
      OWN 70/15/15 re-split of the full recombined, chronologically-sorted
      dataset, matching liar2-split-and-input-ablation-experiments*.ipynb
      and da-minilm-main-model-v1-0.ipynb. This is the split your reported
      DA-MiniLM-RWAC / MiniLM-LogReg-RW numbers were actually evaluated on.

These are NOT the same partition (verified: the official split is
18369/2297/2296 = ~80/10/10; the canonical re-split ratios are 70/15/15
over the recombined data). Don't mix them up when reporting results.

Usage
-----
    from liar2_preprocessor import Liar2Preprocessor

    p = Liar2Preprocessor()

    # step by step, so you can inspect each stage:
    df = p.load_raw(train_path, valid_path, test_path)
    df = p.clean_statement_text(df)
    df = p.normalize_labels(df)
    df = p.filter_invalid_rows(df)
    df = p.parse_and_sort_by_date(df)
    df = p.build_text_modes(df)
    df = p.drop_leakage_columns(df)
    df = p.add_time_rank(df)

    # or all at once:
    df = p.run_pipeline(train_path, valid_path, test_path)

    # re-splits, computed on the fully-cleaned df:
    train_idx, valid_idx, test_idx = p.chronological_split(df)
    train_idx, valid_idx, test_idx = p.random_stratified_split(df, seed=42)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Liar2Preprocessor:

    LABEL_ORDER = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    NUMERIC_LABEL_MAP = {0: "pants-fire", 1: "false", 2: "barely-true",
                         3: "half-true", 4: "mostly-true", 5: "true"}
    # Included in model text (restricted metadata, not leakage-risk):
    SAFE_METADATA_COLUMNS = ["subject", "speaker", "speaker_description", "state_info", "context"]
    # Dropped entirely -- justification and speaker credibility-history counts are leakage risks:
    EXCLUDED_COLUMNS = ["justification", "true_counts", "mostly_true_counts", "half_true_counts",
                        "mostly_false_counts", "false_counts", "pants_on_fire_counts"]

    def __init__(
        self,
        train_ratio: float = 0.70,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        date_coverage_threshold: float = 0.80,
        seed: int = 42,
    ):
        assert abs((train_ratio + valid_ratio + test_ratio) - 1.0) < 1e-9, \
            "train_ratio + valid_ratio + test_ratio must sum to 1.0"
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.date_coverage_threshold = date_coverage_threshold
        self.seed = seed
        self.label_to_id = {label: i for i, label in enumerate(self.LABEL_ORDER)}

    # ================================================================
    # Step 1: Load, tagging each row with its OFFICIAL file-based split
    # ================================================================

    def load_raw(self, train_path, valid_path, test_path) -> pd.DataFrame:
        """Load the three official LIAR2 CSVs and concatenate them, tagging
        each row with `official_split` = "train"/"valid"/"test" based on
        which file it came from. Nothing else is cleaned yet -- this is
        the raw concatenation, useful for auditing raw label distributions
        or row counts per official split before any cleaning happens.
        """
        frames = []
        for path, split_name in [(train_path, "train"), (valid_path, "valid"), (test_path, "test")]:
            part = pd.read_csv(path)
            part["official_split"] = split_name
            part["source_file"] = Path(path).name
            frames.append(part)
        df = pd.concat(frames, ignore_index=True)

        print("Loaded raw LIAR2 rows by official_split:")
        print(df["official_split"].value_counts().reindex(["train", "valid", "test"]))
        return df

    # ================================================================
    # Step 2: Text cleaning
    # ================================================================

    @staticmethod
    def clean_text(x) -> str:
        if pd.isna(x):
            return ""
        return re.sub(r"\s+", " ", str(x)).strip()

    def clean_statement_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Whitespace-collapse and strip the `statement` column."""
        df = df.copy()
        df["statement"] = df["statement"].apply(self.clean_text)
        return df

    # ================================================================
    # Step 3: Label normalization
    # ================================================================

    def _normalize_single_label(self, x):
        if pd.isna(x):
            return np.nan
        try:
            if str(x).strip().replace(".0", "").isdigit():
                xi = int(float(x))
                if xi in self.NUMERIC_LABEL_MAP:
                    return self.NUMERIC_LABEL_MAP[xi]
        except Exception:
            pass
        x = str(x).strip().lower().replace("_", "-")
        x = re.sub(r"\s+", "-", x)
        mapping = {
            "pants-fire": "pants-fire", "pants-on-fire": "pants-fire", "pantsonfire": "pants-fire",
            "false": "false",
            "barely-true": "barely-true", "barelytrue": "barely-true",
            "half-true": "half-true", "halftrue": "half-true",
            "mostly-true": "mostly-true", "mostlytrue": "mostly-true",
            "true": "true",
        }
        return mapping.get(x, x)

    def normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw label (numeric 0-5 or text, with spelling variants) to
        the canonical 6-class label string. Invalid/unmapped labels are
        left as-is here -- use filter_invalid_rows() to drop them."""
        df = df.copy()
        df["label"] = df["label"].apply(self._normalize_single_label)
        return df

    # ================================================================
    # Step 4: Drop empty statements / unmapped labels, assign label_id
    # ================================================================

    def filter_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with an empty statement or a label outside LABEL_ORDER,
        then assign the integer `label_id`. Prints how many rows were
        removed at each stage so you can see exactly what got dropped."""
        n_start = len(df)

        df = df[df["statement"].str.len() > 0].copy()
        n_after_empty = len(df)

        df = df[df["label"].isin(self.LABEL_ORDER)].copy()
        n_after_label = len(df)

        df["label_id"] = df["label"].map(self.label_to_id).astype(int)

        print(f"filter_invalid_rows: {n_start} -> {n_after_empty} "
              f"(dropped {n_start - n_after_empty} empty statements) -> "
              f"{n_after_label} (dropped {n_after_empty - n_after_label} unmappable labels)")
        return df.reset_index(drop=True)

    # ================================================================
    # Step 5: Date parsing + chronological sort (with coverage fallback)
    # ================================================================

    def parse_and_sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse `date` into `date_parsed`. If more than
        `date_coverage_threshold` fraction of rows have a valid date,
        sort the whole frame chronologically; otherwise fall back to the
        original row order (matches the canonical notebooks' behavior).
        Assigns `claim_id` as the post-sort row index either way.
        Sets `df.attrs["sort_source"]` and `df.attrs["date_coverage"]`.
        """
        df = df.copy()
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.NaT

        n_total = len(df)
        n_dated = int(df["date_parsed"].notna().sum())
        coverage = n_dated / n_total if n_total else 0.0

        if coverage > self.date_coverage_threshold:
            df = df.sort_values("date_parsed").reset_index(drop=True)
            sort_source = "date"
        else:
            df = df.reset_index(drop=True)
            sort_source = "row_order"

        df["claim_id"] = np.arange(len(df))
        df.attrs["sort_source"] = sort_source
        df.attrs["date_coverage"] = coverage

        print(f"parse_and_sort_by_date: {n_dated}/{n_total} rows have a valid date "
              f"({coverage:.2%}) -> sort_source = '{sort_source}'")
        return df

    # ================================================================
    # Step 6: Build both text-input modes (for the claim-only vs
    # claim+metadata ablation)
    # ================================================================

    def _build_claim_only(self, row) -> str:
        return f"claim: {row['statement']}"

    def _build_claim_plus_safe_metadata(self, row) -> str:
        parts = [f"claim: {row['statement']}"]
        for col in self.SAFE_METADATA_COLUMNS:
            if col in row.index:
                val = self.clean_text(row.get(col, ""))
                if val:
                    parts.append(f"{col}: {val}")
        return " [SEP] ".join(parts)

    def build_text_modes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds two columns:
            model_text__claim_only
            model_text__claim_plus_safe_metadata
        so both conditions of the text-input ablation are available at
        once without re-running preprocessing."""
        df = df.copy()
        df["model_text__claim_only"] = df.apply(self._build_claim_only, axis=1)
        df["model_text__claim_plus_safe_metadata"] = df.apply(self._build_claim_plus_safe_metadata, axis=1)
        return df

    # ================================================================
    # Step 7: Drop leakage-risk columns entirely
    # ================================================================

    def drop_leakage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops `justification` and all speaker credibility-history count
        columns entirely (not just excludes them from model text), so they
        can't be accidentally used downstream."""
        present = [c for c in self.EXCLUDED_COLUMNS if c in df.columns]
        df = df.drop(columns=present)
        print(f"drop_leakage_columns: removed {present if present else '(none present)'}")
        return df

    # ================================================================
    # Duplicate / conflicting-label handling (opt-in, NOT called by
    # run_pipeline automatically -- see note below)
    # ================================================================

    @staticmethod
    def _normalize_for_overlap(x: str) -> str:
        """Aggressive normalization used ONLY for duplicate detection,
        never for the text fed to a model."""
        x = str(x).lower()
        x = re.sub(r"[^a-z0-9\s]", "", x)
        return re.sub(r"\s+", " ", x).strip()

    def audit_duplicate_statements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Diagnostic only -- does NOT modify df. Groups rows by normalized
        statement text and reports, per group: row count, distinct labels
        present, and whether the group has a genuine label conflict (same
        claim, different verdicts)."""
        work = df.copy()
        work["_norm"] = work["statement"].apply(self._normalize_for_overlap)

        grouped = work.groupby("_norm").agg(
            n_rows=("label", "size"),
            distinct_labels=("label", lambda s: sorted(set(s))),
        ).reset_index()
        grouped["is_duplicate_group"] = grouped["n_rows"] > 1
        grouped["has_label_conflict"] = grouped["distinct_labels"].apply(len) > 1
        return grouped

    def identify_conflicting_label_groups(self, audit_df: pd.DataFrame) -> pd.DataFrame:
        """Subset of audit_duplicate_statements() output where the same
        normalized claim carries more than one distinct label."""
        return audit_df[audit_df["has_label_conflict"]].copy()

    def drop_duplicate_statements(
        self,
        df: pd.DataFrame,
        policy: str = "keep_first",
        audit_df=None,
    ) -> pd.DataFrame:
        """
        This method is also called by the canonical LIAR2 preprocessing
        pipeline in this revised notebook. The pipeline uses policy="keep_first"
        so every normalized-statement duplicate group contributes only its
        chronologically earliest row.

        policy:
            "keep_first"              -> for every duplicate group (exact
                                          or conflicting-label), keep only
                                          the chronologically earliest row.
            "drop_conflicting_groups" -> conflicting-label groups are
                                          DROPPED ENTIRELY (all rows in the
                                          group removed); non-conflicting
                                          duplicate groups still keep the
                                          earliest row.
        """
        if audit_df is None:
            audit_df = self.audit_duplicate_statements(df)

        work = df.copy()
        work["_norm"] = work["statement"].apply(self._normalize_for_overlap)

        conflicts = self.identify_conflicting_label_groups(audit_df)
        n_conflict_groups = len(conflicts)
        n_conflict_rows = int(work["_norm"].isin(conflicts["_norm"]).sum())
        if n_conflict_groups > 0:
            print(f"drop_duplicate_statements: {n_conflict_groups} conflicting-label group(s) "
                  f"covering {n_conflict_rows} row(s) found.")

        if policy == "drop_conflicting_groups" and n_conflict_groups > 0:
            work = work[~work["_norm"].isin(conflicts["_norm"])].copy()

        before = len(work)
        sort_col = "date_parsed" if "date_parsed" in work.columns else "claim_id"
        work = work.sort_values(sort_col).drop_duplicates(subset=["_norm"], keep="first")
        after = len(work)
        print(f"drop_duplicate_statements ({policy}): {before} -> {after} (removed {before - after})")

        return work.drop(columns=["_norm"]).reset_index(drop=True)

    # ================================================================
    # Step 8: Time features (for Time2Vec / domain-adversarial component)
    # ================================================================

    def add_time_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds `time_rank` in [0, 1], a global linspace over the full
        chronologically-sorted frame. Must be called AFTER
        parse_and_sort_by_date(). Matches
        liar2-split-and-input-ablation-experiments3.ipynb and the main
        model notebook exactly (global rank, computed before any re-split)."""
        df = df.copy()
        df["time_rank"] = np.linspace(0.0, 1.0, len(df))
        return df

    def make_domain_buckets(self, time_rank_train: np.ndarray, num_domains: int = 5) -> np.ndarray:
        """Quantile-bucket TRAIN rows' time_rank into `num_domains` discrete
        ids, for domain-adversarial training. Pass only train-split
        time_rank values -- never valid/test -- to avoid leakage into the
        domain-bucket boundaries."""
        quantiles = np.quantile(time_rank_train, np.linspace(0, 1, num_domains + 1))
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        return np.digitize(time_rank_train, quantiles[1:-1], right=True)

    # ================================================================
    # Re-splits (project's own 70/15/15 protocol, distinct from
    # official_split)
    # ================================================================

    def chronological_split(self, df: pd.DataFrame):
        """Contiguous 70/15/15 split in row order. Requires df to already
        be chronologically sorted (i.e. call parse_and_sort_by_date() first
        with sort_source == 'date')."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        valid_end = int(n * (self.train_ratio + self.valid_ratio))
        return (
            np.arange(0, train_end),
            np.arange(train_end, valid_end),
            np.arange(valid_end, n),
        )

    def random_stratified_split(self, df: pd.DataFrame, seed: int = None):
        """70/15/15 stratified-by-label split, re-sorted chronologically
        within each split so recency weighting remains meaningful."""
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

    def get_split(self, df: pd.DataFrame, split_mode: str, seed: int = None):
        """Added so Liar2Preprocessor is genuinely interchangeable with
        TemporalSplitter wherever a `.splitter` is expected (e.g.
        DAMiniLMExperimentRunner.run_full_battery()) -- confirmed missing
        via a real crash (AttributeError) when first wiring the two
        together in a full end-to-end run; not just a hypothetical gap."""
        if split_mode == "chronological":
            return self.chronological_split(df)
        if split_mode == "random_stratified":
            return self.random_stratified_split(df, seed=seed)
        raise ValueError(f"Unknown split_mode: {split_mode}")

    def make_rolling_windows(
        self,
        df: pd.DataFrame,
        min_train_frac: float = 0.50,
        valid_frac: float = 0.10,
        test_frac: float = 0.10,
        step_frac: float = 0.10,
    ) -> list:
        """Expanding-window temporal evaluation windows, for the temporal-
        generalization / rolling-window diagnostics."""
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
                "window_id": window_id,
                "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx,
                # These three were missing until a real end-to-end dry run
                # crashed inside DAMiniLMExperimentRunner.run_rolling_window_evaluation(),
                # which expects the same keys TemporalSplitter.make_rolling_windows()
                # returns -- added here so the two are genuinely interchangeable,
                # not just interchangeable for chronological_split/random_stratified_split.
                "train_end_frac": train_end / n, "valid_end_frac": valid_end / n, "test_end_frac": test_end / n,
                "n_train": len(train_idx), "n_valid": len(valid_idx), "n_test": len(test_idx),
            })
            train_end += step
            window_id += 1
        return windows

    # ================================================================
    # Full pipeline (runs every step above, in order)
    # ================================================================

    def run_pipeline(self, train_path, valid_path, test_path) -> pd.DataFrame:
        """Runs the canonical LIAR2 preprocessing sequence used by this notebook:
            load_raw -> clean_statement_text -> normalize_labels ->
            filter_invalid_rows -> parse_and_sort_by_date ->
            build_text_modes -> drop_leakage_columns ->
            audit/drop normalized-statement duplicates (keep_first) ->
            chronological re-sort -> add_time_rank

        The official LIAR2 files contain 22,962 rows. Under the current
        dataset version and duplicate-normalization rule, the expected final
        analysis cohort is 22,923 unique normalized statements.
        """
        df = self.load_raw(train_path, valid_path, test_path)
        df = self.clean_statement_text(df)
        df = self.normalize_labels(df)
        df = self.filter_invalid_rows(df)
        df = self.parse_and_sort_by_date(df)
        df = self.build_text_modes(df)
        df = self.drop_leakage_columns(df)

        audit = self.audit_duplicate_statements(df)
        df = self.drop_duplicate_statements(df, policy="keep_first", audit_df=audit)

        # Row removal changes the global temporal rank. Re-sort and recompute it
        # on the final cohort before any split is constructed.
        df = self.parse_and_sort_by_date(df)
        df = self.add_time_rank(df).reset_index(drop=True)

        if len(df) != 22923:
            raise ValueError(
                f"Expected final LIAR2 cohort of 22,923 rows after deduplication; got {len(df):,}. "
                "Check whether the dataset version or duplicate-normalization rule changed."
            )

        print(f"\nrun_pipeline complete: {len(df)} FINAL DEDUPLICATED rows.")
        print("Official-split provenance counts within the final cohort:")
        print(df["official_split"].value_counts().reindex(["train", "valid", "test"]))
        return df


if __name__ == "__main__":
    print("liar2_preprocessor.py loaded. Class: Liar2Preprocessor")