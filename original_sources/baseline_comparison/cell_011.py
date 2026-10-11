"""
multifc_preprocessor.py

One class, MultiFC only -- structured the same way as
Liar2Preprocessor/LiarPreprocessor (every step its own method, official
split preserved, leakage-safe metadata). MultiFC is this project's
external-replication dataset: per project methodology, it is evaluated as
INDEPENDENT RETRAINING on MultiFC's own official train/dev/test split,
NOT as zero-shot transfer from a LIAR2-trained model. There is no
chronological split for MultiFC in this class -- retraining uses the
official split directly.

MultiFC's raw TSVs are the 37-site fact-checking corpus (Wang, 2017,
Constructing Datasets for Multi-domain Truth Assessment) -- 13 columns
when labelled (train/dev), 12 when not (test). Its ~26+ site-specific
label taxonomy is harmonized here to 3 classes (false/mixed/true),
matching this project's established label-collapse decisions:
    - ~13-19% of rows are typically dropped as unmappable/unlabelled --
      this must be reported, not silently absorbed (see
      export_attrition_report()).
    - `source_site` (parsed from claim_id's alphabetic prefix) is NOT
      bundled into the default restricted-metadata text -- it's a
      plausible site-identity shortcut and is exposed as its own,
      separate text mode instead, so using it is a deliberate choice,
      not an unquestioned default.
    - `title` and `evidence_text` are excluded from all default text
      modes -- both can directly leak the verdict.

Usage
-----
    from multifc_preprocessor import MultiFCPreprocessor

    p = MultiFCPreprocessor()
    df = p.run_pipeline(train_path, dev_path, test_path)
    p.export_attrition_report("./multifc_reports")
"""

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


LABELLED_COLUMNS = [
    "claim_id", "claim", "raw_label", "url", "evidence_text",
    "category", "speaker", "checker", "tags", "title",
    "published_date", "claim_date", "entities",
]
UNLABELLED_COLUMNS = [
    "claim_id", "claim", "url", "evidence_text",
    "category", "speaker", "checker", "tags", "title",
    "published_date", "claim_date", "entities",
]


class MultiFCPreprocessor:

    LABEL_ORDER = ["false", "mixed", "true"]
    # Included in model text (restricted metadata, not leakage-risk):
    SAFE_METADATA_COLUMNS = ["category", "speaker", "entities"]
    # NOT included by default -- plausible site-identity shortcut, opt-in only:
    SOURCE_SITE_COLUMN = "source_site"
    # NEVER included -- can directly leak the verdict:
    EXCLUDED_COLUMNS = ["title", "evidence_text", "url", "checker", "tags"]

    def __init__(self):
        self.label_to_id = {label: i for i, label in enumerate(self.LABEL_ORDER)}
        self._build_label_taxonomy()
        self.attrition_report_: dict = {}
        self.unmapped_labels_: dict = {}
        self.class_distribution_: dict = {}

    # ================================================================
    # Label taxonomy (26-site -> 3-class harmonization)
    # ================================================================

    def _build_label_taxonomy(self):
        def norm(x):
            return re.sub(r"\s+", " ", str(x).strip().lower().replace("_", "-"))

        self._true_labels = set(norm(x) for x in [
            "true", "truth!", "verdict: true", "determination: true", "factscan score: true",
            "correct", "accurate", "verified", "fact", "conclusion: accurate",
            "mostly true", "mostly-true", "mostly_true", "mostly truth!", "mostly-correct",
            "determination: mostly true", "promise kept", "in-the-green", "true messages",
            "authorship confirmed!", "confirmed authorship!", "correct attribution",
            "correct attribution!", "truth! & outdated!", "truth! & unproven!",
            "previously truth! now resolved!",
        ])
        self._mixed_labels = set(norm(x) for x in [
            "half-true", "half true", "mixture", "in-between", "partly true", "partially true",
            "needs context", "not the whole story", "compromise", "determination: barely true",
            "truth! & fiction!", "truth! & misleading!", "truth! & disputed!", "disputed!",
            "unproven", "unproven!", "unverified", "determination: misleading",
            "factscan score: misleading", "misleading", "misleading!", "misleading recommendations",
            "2 pinnochios", "a little baloney", "some baloney", "spins the facts",
            "distorts the facts", "cherry picks", "exaggerated", "exaggerates", "understated",
            "half flip", "no flip", "conclusion: unclear", "determination: a stretch",
        ])
        self._false_labels = set(norm(x) for x in [
            "false", "mostly false", "mostly-false", "mostly_false", "pants on fire!",
            "pants-fire", "fiction!", "fiction", "mostly fiction!", "verdict: false",
            "determination: false", "factscan score: false", "conclusion: false",
            "rating: false", "we rate this claim false", "incorrect", "incorrect attribution!",
            "inaccurate attribution!", "fake", "fake news",
        ])
        # Explicitly-known non-mappable categories (distinct from "unrecognized").
        self._drop_labels = set(norm(x) for x in [
            "legend", "satire", "fiction! & satire!", "statirical reports",
            "determination: huckster propaganda",
        ])

    def harmonize_label(self, raw_label) -> Optional[str]:
        """Returns 'true'/'mixed'/'false', or None if unmappable (either an
        explicit drop category, a bare numeric code, or genuinely
        unrecognized). Consult self.unmapped_labels_ after clean_and_filter()
        to distinguish these for reporting."""
        if pd.isna(raw_label):
            return None
        norm = re.sub(r"\s+", " ", str(raw_label).strip().lower().replace("_", "-"))
        if norm in self._true_labels:
            return "true"
        if norm in self._mixed_labels:
            return "mixed"
        if norm in self._false_labels:
            return "false"
        if norm in self._drop_labels:
            return None
        if re.fullmatch(r"\d+", norm):
            return None
        return None

    # ================================================================
    # Step 1: Load, tagging each row with its official split
    # ================================================================

    @staticmethod
    def clean_text(x) -> str:
        """MultiFC's raw TSVs use the literal string "None" (not NaN) for
        empty fields -- treated as empty here, matching what real data
        inspection confirmed."""
        if pd.isna(x):
            return ""
        x = re.sub(r"\s+", " ", str(x)).strip()
        if x.lower() == "none":
            return ""
        return x

    @staticmethod
    def parse_entities(x) -> str:
        """MultiFC's raw entities column is a stringified Python list, e.g.
        "['None']" (no entities) or "['United_States', 'Barack_Obama']".
        Extracts real entity names, dropping the 'None' placeholder."""
        if pd.isna(x):
            return ""
        items = re.findall(r"'([^']*)'", str(x))
        items = [i for i in items if i and i.lower() != "none"]
        return ", ".join(items)

    def _read_tsv(self, path, split_name: str) -> pd.DataFrame:
        raw = pd.read_csv(path, sep="\t", header=None, dtype=str,
                           keep_default_na=False, quoting=3, engine="python")
        if raw.shape[1] == 13:
            raw.columns = LABELLED_COLUMNS
        elif raw.shape[1] == 12:
            raw.columns = UNLABELLED_COLUMNS
            raw["raw_label"] = np.nan
        else:
            raise ValueError(f"{path} has {raw.shape[1]} columns, expected 12 or 13.")
        raw["official_split"] = split_name
        raw["source_file"] = Path(path).name
        raw[self.SOURCE_SITE_COLUMN] = (
            raw["claim_id"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False).fillna("unknown")
        )
        return raw

    def load_raw(self, train_path, dev_path, test_path) -> pd.DataFrame:
        """Load MultiFC's three official TSVs and concatenate, tagging
        each row with `official_split`. Nothing else is cleaned yet."""
        frames = [
            self._read_tsv(train_path, "train"),
            self._read_tsv(dev_path, "dev"),
            self._read_tsv(test_path, "test"),
        ]
        df = pd.concat(frames, ignore_index=True)
        print("Loaded raw MultiFC rows by official_split:")
        print(df["official_split"].value_counts().reindex(["train", "dev", "test"]))
        return df

    # ================================================================
    # Step 2: Text cleaning
    # ================================================================

    def clean_claim_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["claim"] = df["claim"].apply(self.clean_text)
        return df

    # ================================================================
    # Step 3: Label harmonization
    # ================================================================

    def normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps raw_label -> harmonized 3-class label. Tracks unmapped
        raw labels per split for later reporting (export_attrition_report)."""
        df = df.copy()
        df["label"] = df["raw_label"].apply(self.harmonize_label)

        for split_name in df["official_split"].unique():
            mask = df["official_split"] == split_name
            unmapped_mask = mask & df["raw_label"].notna() & df["label"].isna()
            counts = df.loc[unmapped_mask, "raw_label"].value_counts()
            if len(counts):
                self.unmapped_labels_[split_name] = counts

        return df

    # ================================================================
    # Step 4: Drop empty claims / unmapped labels, assign label_id
    # ================================================================

    def filter_invalid_rows(self, df: pd.DataFrame, drop_unlabelled: bool = False) -> pd.DataFrame:
        """Drops rows with an empty claim. `drop_unlabelled` controls
        whether rows with no mappable label are also dropped here --
        default False, since the officially unlabelled test split should
        usually be KEPT (with label=NaN) for held-out submission-style
        evaluation, and dropped only for splits you intend to train/
        validate on. Call with drop_unlabelled=True on train/dev
        specifically if you want a train/dev/test-ready DataFrame.
        Records per-split attrition either way."""
        n_before = len(df)
        df = df[df["claim"].str.len() > 0].copy()
        n_after_empty = len(df)

        if drop_unlabelled:
            df = df[df["label"].isin(self.LABEL_ORDER)].copy()
        n_after_label = len(df)

        df["label_id"] = df["label"].map(self.label_to_id)

        for split_name in df["official_split"].unique():
            split_df = df[df["official_split"] == split_name]
            n_split = len(split_df)
            n_labelled = int(split_df["label"].isin(self.LABEL_ORDER).sum())
            dropped_frac = 1 - (n_labelled / n_split) if n_split else 0.0
            self.attrition_report_[split_name] = {
                "n_rows": int(n_split), "n_with_mappable_label": n_labelled,
                "dropped_unmappable_frac": round(dropped_frac, 4),
            }
            print(f"MultiFC {split_name}: {n_labelled}/{n_split} rows have a mappable label "
                  f"({dropped_frac:.1%} unmappable/unlabelled)")

        print(f"filter_invalid_rows: {n_before} -> {n_after_empty} empty-claim rows removed"
              + (f" -> {n_after_label} after dropping unlabelled" if drop_unlabelled else ""))
        return df.reset_index(drop=True)

    # ================================================================
    # Step 5: Build text modes
    # ================================================================

    def _build_claim_only(self, row) -> str:
        return self.clean_text(row.get("claim", ""))

    def _build_claim_plus_safe_metadata(self, row) -> str:
        parts = [self.clean_text(row.get("claim", ""))]
        for col in ("category", "speaker"):
            val = self.clean_text(row.get(col, ""))
            if val:
                parts.append(f"{col}: {val}")
        entities_val = self.parse_entities(row.get("entities", ""))
        if entities_val:
            parts.append(f"entities: {entities_val}")
        return " [SEP] ".join(p for p in parts if p)

    def _build_claim_plus_safe_metadata_and_source(self, row) -> str:
        base = self._build_claim_plus_safe_metadata(row)
        source_val = self.clean_text(row.get(self.SOURCE_SITE_COLUMN, ""))
        if source_val:
            return f"{base} [SEP] source: {source_val}"
        return base

    def build_text_modes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds:
            model_text__claim_only
            model_text__claim_plus_safe_metadata          (category, speaker, entities)
            model_text__claim_plus_safe_metadata_and_source (+ source_site -- opt-in
                                                              sensitivity-analysis mode,
                                                              a plausible site-identity
                                                              shortcut)
        """
        df = df.copy()
        df["model_text__claim_only"] = df.apply(self._build_claim_only, axis=1)
        df["model_text__claim_plus_safe_metadata"] = df.apply(self._build_claim_plus_safe_metadata, axis=1)
        df["model_text__claim_plus_safe_metadata_and_source"] = df.apply(
            self._build_claim_plus_safe_metadata_and_source, axis=1,
        )
        return df

    # ================================================================
    # Step 6: Drop leakage-risk columns entirely
    # ================================================================

    def drop_leakage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops title/evidence_text/url/checker/tags entirely (not just
        excludes them from model text)."""
        present = [c for c in self.EXCLUDED_COLUMNS if c in df.columns]
        df = df.drop(columns=present)
        print(f"drop_leakage_columns: removed {present if present else '(none present)'}")
        return df

    # ================================================================
    # Reporting
    # ================================================================

    def export_attrition_report(self, output_dir) -> None:
        """Writes:
            multifc_attrition_by_split.csv
            multifc_unmapped_labels.csv
            multifc_class_distribution.csv
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(self.attrition_report_).T.to_csv(output_dir / "multifc_attrition_by_split.csv")

        unmapped_rows = []
        for split_name, counts in self.unmapped_labels_.items():
            for raw_label, n in counts.items():
                unmapped_rows.append({"split": split_name, "raw_label": raw_label, "count": n})
        pd.DataFrame(unmapped_rows).to_csv(output_dir / "multifc_unmapped_labels.csv", index=False)

        dist_rows = []
        for stage, dist in self.class_distribution_.items():
            for label, n in dist.items():
                dist_rows.append({"stage": stage, "label": label, "count": n})
        pd.DataFrame(dist_rows).to_csv(output_dir / "multifc_class_distribution.csv", index=False)

        print(f"Wrote attrition/unmapped-label/class-distribution reports to {output_dir}")

    # ================================================================
    # Official split retrieval (no chronological split for MultiFC --
    # independent retraining uses the official split directly)
    # ================================================================

    def get_official_split(self, df: pd.DataFrame):
        """Returns (train_df, dev_df, test_df) using official_split.
        train_df/dev_df are filtered to rows with a mappable label
        (label_id not null); test_df is returned as-is (MultiFC's
        official test partition is unlabelled for the shared-task
        setting -- keep unlabelled rows if you need them, or pre-filter
        yourself if you have gold labels for test)."""
        train_df = df[(df["official_split"] == "train") & df["label"].isin(self.LABEL_ORDER)].copy()
        dev_df = df[(df["official_split"] == "dev") & df["label"].isin(self.LABEL_ORDER)].copy()
        test_df = df[df["official_split"] == "test"].copy()
        return train_df, dev_df, test_df

    # ================================================================
    # Full pipeline
    # ================================================================

    def run_pipeline(self, train_path, dev_path, test_path, drop_unlabelled: bool = False) -> pd.DataFrame:
        """load_raw -> clean_claim_text -> normalize_labels ->
        filter_invalid_rows -> build_text_modes -> drop_leakage_columns"""
        df = self.load_raw(train_path, dev_path, test_path)
        df = self.clean_claim_text(df)
        df = self.normalize_labels(df)
        df = self.filter_invalid_rows(df, drop_unlabelled=drop_unlabelled)
        df = self.build_text_modes(df)
        df = self.drop_leakage_columns(df)

        for split_name in ("train", "dev"):
            split_df = df[df["official_split"] == split_name]
            self.class_distribution_[f"{split_name}_after_filter"] = (
                split_df["label"].value_counts().to_dict()
            )

        print(f"\nrun_pipeline complete: {len(df)} final rows.")
        return df


if __name__ == "__main__":
    print("multifc_preprocessor.py loaded. Class: MultiFCPreprocessor")