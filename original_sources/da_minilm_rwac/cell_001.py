"""
liar_preprocessor.py

One class, original LIAR dataset only (Wang, 2017) -- structured the same
way as Liar2Preprocessor (every step its own method, official_split
preserved, opt-in dedup/conflict auditing), with ONE deliberate structural
difference: LIAR has no date column, so nothing here does date parsing,
chronological sorting, time_rank, domain buckets, or rolling windows.
Only a 70/15/15 RANDOM STRATIFIED re-split is offered alongside the
official file-based split -- there is no "chronological_split" method on
this class, and there cannot be one without a date field to sort by.

LIAR's raw TSV format (no header row) has 14 columns, in this order:
    id, label, statement, subject, speaker, job_title, state_info,
    party_affiliation, barely_true_count, false_count, half_true_count,
    mostly_true_count, pants_on_fire_count, context
Confirmed against the actual uploaded train.tsv/valid.tsv/test.tsv.

Note LIAR's schema differs from LIAR2's in a few ways relevant to
preprocessing:
    - No `date` column at all (LIAR2 has one) -> no time-based anything.
    - No `justification` column (LIAR2 has one) -> nothing to exclude there.
    - No `speaker_description` column (LIAR2 has one).
    - Has `job_title` and `party_affiliation` columns that LIAR2 does not.
    - Credibility-history counts use different column names but the same
      underlying concept (barely_true_count, false_count, half_true_count,
      mostly_true_count, pants_on_fire_count) -- these are excluded from
      model text and dropped entirely, for the same leakage reason as in
      Liar2Preprocessor.

Usage
-----
    from liar_preprocessor import LiarPreprocessor

    p = LiarPreprocessor()

    df = p.load_raw(train_path, valid_path, test_path)
    df = p.clean_statement_text(df)
    df = p.normalize_labels(df)
    df = p.filter_invalid_rows(df)
    df = p.build_text_modes(df)
    df = p.drop_leakage_columns(df)

    # or all at once:
    df = p.run_pipeline(train_path, valid_path, test_path)

    # only re-split available (no chronological_split -- no dates exist):
    train_idx, valid_idx, test_idx = p.random_stratified_split(df, seed=42)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class LiarPreprocessor:

    RAW_COLUMNS = [
        "id", "label", "statement", "subject", "speaker", "job_title",
        "state_info", "party_affiliation", "barely_true_count", "false_count",
        "half_true_count", "mostly_true_count", "pants_on_fire_count", "context",
    ]

    LABEL_ORDER = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    NUMERIC_LABEL_MAP = {0: "pants-fire", 1: "false", 2: "barely-true",
                         3: "half-true", 4: "mostly-true", 5: "true"}
    # Included in model text (restricted metadata, not leakage-risk):
    SAFE_METADATA_COLUMNS = ["subject", "speaker", "job_title", "state_info",
                             "party_affiliation", "context"]
    # Dropped entirely -- speaker credibility-history counts are leakage risks:
    EXCLUDED_COLUMNS = ["barely_true_count", "false_count", "half_true_count",
                        "mostly_true_count", "pants_on_fire_count"]

    def __init__(
        self,
        train_ratio: float = 0.70,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        assert abs((train_ratio + valid_ratio + test_ratio) - 1.0) < 1e-9, \
            "train_ratio + valid_ratio + test_ratio must sum to 1.0"
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.label_to_id = {label: i for i, label in enumerate(self.LABEL_ORDER)}

    # ================================================================
    # Step 1: Load, tagging each row with its OFFICIAL file-based split
    # ================================================================

    def load_raw(self, train_path, valid_path, test_path) -> pd.DataFrame:
        """Load the three official LIAR TSVs (no header row in the raw
        files) and concatenate them, tagging each row with
        `official_split` = "train"/"valid"/"test" based on which file it
        came from. `row_order` records original concatenation order --
        used later as the tie-break for duplicate removal, since LIAR has
        no date column to sort by.
        """
        frames = []
        for path, split_name in [(train_path, "train"), (valid_path, "valid"), (test_path, "test")]:
            part = pd.read_csv(path, sep="\t", header=None, names=self.RAW_COLUMNS, dtype=str)
            part["official_split"] = split_name
            part["source_file"] = Path(path).name
            frames.append(part)
        df = pd.concat(frames, ignore_index=True)
        df["row_order"] = np.arange(len(df))

        print("Loaded raw LIAR rows by official_split:")
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
        """Map raw label (LIAR's labels are already text, but numeric 0-5
        is also handled for consistency with LiarPreprocessor's sibling
        class) to the canonical 6-class label string. Invalid/unmapped
        labels are left as-is here -- use filter_invalid_rows() to drop
        them."""
        df = df.copy()
        df["label"] = df["label"].apply(self._normalize_single_label)
        return df

    # ================================================================
    # Step 4: Drop empty statements / unmapped labels, assign label_id
    # ================================================================

    def filter_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with an empty statement or a label outside
        LABEL_ORDER, then assign the integer `label_id`. Prints how many
        rows were removed at each stage."""
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
    # Step 5: Build both text-input modes
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
        once."""
        df = df.copy()
        df["model_text__claim_only"] = df.apply(self._build_claim_only, axis=1)
        df["model_text__claim_plus_safe_metadata"] = df.apply(self._build_claim_plus_safe_metadata, axis=1)
        return df

    # ================================================================
    # Step 6: Drop leakage-risk columns entirely
    # ================================================================

    def drop_leakage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops the speaker credibility-history count columns entirely
        (not just excludes them from model text), so they can't be
        accidentally used downstream. (LIAR has no `justification` column
        to drop -- that's LIAR2-specific.)"""
        present = [c for c in self.EXCLUDED_COLUMNS if c in df.columns]
        df = df.drop(columns=present)
        print(f"drop_leakage_columns: removed {present if present else '(none present)'}")
        return df

    # ================================================================
    # Duplicate / conflicting-label handling (opt-in, NOT called by
    # run_pipeline automatically)
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
        present, and whether the group has a genuine label conflict."""
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
        NOT called automatically by run_pipeline -- opt-in only, since it
        changes row count. Since LIAR has no date column, "earliest" here
        means earliest by `row_order` (original file concatenation order:
        train rows before valid rows before test rows) rather than by
        timestamp.

        policy:
            "keep_first"              -> for every duplicate group (exact
                                          or conflicting-label), keep only
                                          the row with the lowest row_order.
            "drop_conflicting_groups" -> conflicting-label groups are
                                          DROPPED ENTIRELY; non-conflicting
                                          duplicate groups still keep the
                                          row_order-earliest row.
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
        sort_col = "row_order" if "row_order" in work.columns else work.index
        work = work.sort_values(sort_col).drop_duplicates(subset=["_norm"], keep="first")
        after = len(work)
        print(f"drop_duplicate_statements ({policy}): {before} -> {after} (removed {before - after})")

        return work.drop(columns=["_norm"]).reset_index(drop=True)

    # ================================================================
    # Re-split (random stratified ONLY -- no chronological_split exists,
    # because LIAR has no date column to sort by)
    # ================================================================

    def random_stratified_split(self, df: pd.DataFrame, seed: int = None):
        """70/15/15 stratified-by-label split. This is the ONLY re-split
        offered on this class -- unlike Liar2Preprocessor, there is no
        chronological_split() here, because LIAR has no date field. If you
        need a temporal analysis, it has to be done on LIAR2, not LIAR."""
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

    # ================================================================
    # Full pipeline
    # ================================================================

    def run_pipeline(self, train_path, valid_path, test_path) -> pd.DataFrame:
        """Runs every step in sequence:
            load_raw -> clean_statement_text -> normalize_labels ->
            filter_invalid_rows -> build_text_modes -> drop_leakage_columns
        No date parsing, no time_rank, no chronological_split step exists
        for this dataset -- see class docstring.
        """
        df = self.load_raw(train_path, valid_path, test_path)
        df = self.clean_statement_text(df)
        df = self.normalize_labels(df)
        df = self.filter_invalid_rows(df)
        df = self.build_text_modes(df)
        df = self.drop_leakage_columns(df)

        print(f"\nrun_pipeline complete: {len(df)} final rows.")
        print("Official split counts (post-cleaning):")
        print(df["official_split"].value_counts().reindex(["train", "valid", "test"]))
        return df


if __name__ == "__main__":
    print("liar_preprocessor.py loaded. Class: LiarPreprocessor "
          "(no chronological_split -- LIAR has no date column)")