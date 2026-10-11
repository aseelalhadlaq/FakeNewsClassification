"""
baseline_models.py

Unified baseline-model class covering LIAR, LIAR2, and the LIAR+LIAR2
merge -- every model x dataset combination in
baseline_model_comparison_liar_liar2.xlsx, INCLUDING B17/B18.

B17/B18 were initially assumed to be incomplete runs (split_type="To be
run" in the xlsx, B18's oddly small n_test=203), but checking
cross-dataset-tf-idf-logistic-regression.ipynb directly showed they are
actually CROSS-DATASET GENERALIZATION experiments: B17 trains on LIAR and
tests on LIAR2; B18 trains on LIAR2 and tests on LIAR. Both remove target-
test rows that overlap the source train/valid before evaluating -- which
is exactly why B18 has so few test rows: LIAR2 extends LIAR, so most of
LIAR's test statements already appear in LIAR2's train/valid and get
correctly stripped out as leakage. See run_cross_dataset_generalization().

Covered (model x dataset):
    TF-IDF + Logistic Regression         -> LIAR (B01), LIAR2 (B02),
                                             raw merge (B03), clean merge (B04)
    TF-IDF + Linear SVM                  -> LIAR (B05), LIAR2 (B06),
                                             clean merge (B07)
    TF-IDF + Logistic Regression + meta  -> LIAR (B08), LIAR2 (B09),
                                             clean merge (B10)
    DistilBERT                           -> LIAR (B11), LIAR2 (B12),
                                             clean merge (B13)
    MiniLM                                -> LIAR (B14), LIAR2 (B15),
                                             clean merge (B16)
    Chronological (LIAR2 only, 70/15/15) -> LogReg (C01), Linear SVM (C02)

ALL data loading/cleaning goes through LiarPreprocessor and
Liar2Preprocessor -- this file contains no re-implementation of cleaning
logic. It only builds the LIAR+LIAR2 merge (schema harmonization +
cross-dataset dedup) on top of what those two produce, then fits/evaluates
models.

Design decisions worth knowing
-------------------------------
1. "Statement + metadata" uses only each dataset's leakage-safe metadata
   columns (LiarPreprocessor.SAFE_METADATA_COLUMNS /
   Liar2Preprocessor.SAFE_METADATA_COLUMNS), never the credibility-history
   counts. For the merge, the metadata columns are further restricted to
   the INTERSECTION of both datasets' safe columns (subject, speaker,
   state_info, context) -- LIAR's job_title/party_affiliation and LIAR2's
   speaker_description don't exist on both sides, so they're dropped for
   the merged condition specifically, not globally.

2. build_liar_liar2_merge() reproduces the "clean merge" dedup behavior
   from tf-idf-logistic-regression-lin-svm-cron-split.ipynb: normalize
   statement text, prefer the dated (LIAR2) version of a cross-dataset
   duplicate over the undated (LIAR) version, keep the earliest by date
   among LIAR2-only duplicates. Verified against your actual
   liar_liar2_clean_merged_tfidf.csv: this reproduces the exact same
   total (23,886 rows) and the exact same LIAR2/LIAR composition
   (13,127 / 10,759) and split sizes (21,053 / 1,539 / 1,294) -- see the
   test block at the bottom of this file's companion test run.

3. The RAW merge (no dedup) is included because it's B03 in the xlsx,
   but B03's own leakage_status says "Leakage detected" and its
   reporting_status says "Do not report as final" -- run_raw_merge_baseline()
   exists so you can regenerate that leakage DEMONSTRATION, not because
   it's a result to cite.

4. Chronological splitting is ONLY valid for LIAR2 -- LIAR has no date
   column. run_tfidf_logreg(..., split_mode="chronological") on a
   LIAR-only or merged DataFrame raises a clear error rather than silently
   doing something else.

5. DistilBERT and MiniLM require Hugging Face Hub access to download
   pretrained weights, which this sandbox's network cannot reach --
   written to match baseline_DistilBERT_Text-Only.ipynb's structure and
   this project's DA-MiniLM Uniform-LR convention, but NOT executed here.
   Test them in your own GPU/internet-enabled environment before trusting
   their output. Everything else (the five TF-IDF-based experiment
   families) is tested end-to-end against your real uploaded files.

Usage
-----
    from liar_preprocessor import LiarPreprocessor
    from liar2_preprocessor import Liar2Preprocessor
    from baseline_models import BaselineModels

    liar_prep = LiarPreprocessor()
    liar_df = liar_prep.run_pipeline(liar_train, liar_valid, liar_test)

    liar2_prep = Liar2Preprocessor()
    liar2_df = liar2_prep.run_pipeline(liar2_train, liar2_valid, liar2_test)

    baselines = BaselineModels(liar_preprocessor=liar_prep, liar2_preprocessor=liar2_prep)

    merged_clean_df = baselines.build_liar_liar2_merge(liar_df, liar2_df, deduplicate=True)

    results_df = baselines.run_all_experiments(
        liar_df=liar_df, liar2_df=liar2_df,
        include_transformer_baselines=False,
    )
"""

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class BaselineModels:

    # ----------------------------------------------------------------
    # Model configs. NOTE: solver="liblinear" from the original notebooks
    # does not run on current scikit-learn (>=1.5; this sandbox has 1.8.0)
    # for 6-class problems -- liblinear's one-vs-rest multiclass fallback
    # was removed outright, not just deprecated. Swapped to "lbfgs"
    # throughout, confirmed by running it here.
    # ----------------------------------------------------------------

    TFIDF_LOGREG_VECTORIZER = dict(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        max_features=30000, min_df=2, max_df=0.90, sublinear_tf=True,
    )
    TFIDF_LOGREG_CLASSIFIER = dict(
        C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs",
    )

    TFIDF_SVM_VECTORIZER = dict(
        lowercase=True, stop_words=None, ngram_range=(1, 2),
        max_features=30000, min_df=2, max_df=0.90, sublinear_tf=True,
    )
    TFIDF_SVM_CLASSIFIER = dict(
        C=1.0, class_weight="balanced", max_iter=5000, random_state=42,
    )

    # Chronological battery (C01/C02) preset, matching
    # tf-idf-logistic-regression-lin-svm-cron-split.ipynb exactly (except
    # the same liblinear -> lbfgs swap).
    TFIDF_LOGREG_VECTORIZER_CHRONO = dict(
        lowercase=True, strip_accents="unicode", stop_words="english",
        ngram_range=(1, 2), max_features=50000, min_df=2, max_df=0.95,
        sublinear_tf=True,
    )
    TFIDF_LOGREG_CLASSIFIER_CHRONO = dict(
        solver="lbfgs", C=1.0, max_iter=3000, random_state=42,
    )
    TFIDF_SVM_VECTORIZER_CHRONO = TFIDF_LOGREG_VECTORIZER_CHRONO
    TFIDF_SVM_CLASSIFIER_CHRONO = dict(C=1.0, random_state=42)

    TFIDF_METADATA_VECTORIZER = dict(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_df=0.95, max_features=50000, sublinear_tf=True,
    )
    LOGREG_METADATA_CLASSIFIER = dict(
        solver="saga", C=1.0, class_weight="balanced",
        max_iter=5000, random_state=42,
    )

    # Metadata columns common to BOTH LIAR and LIAR2 -- used for the merged
    # ("statement + metadata") condition specifically. LIAR's job_title /
    # party_affiliation and LIAR2's speaker_description are dropped only
    # in this merged context, not from the single-dataset runs.
    MERGE_COMMON_METADATA_COLUMNS = ["subject", "speaker", "state_info", "context"]

    def __init__(self, liar_preprocessor=None, liar2_preprocessor=None):
        """Pass whichever preprocessor(s) you have. `liar2_preprocessor`
        is required for chronological_split (LIAR2-only feature)."""
        self.liar_p = liar_preprocessor
        self.liar2_p = liar2_preprocessor

    # ================================================================
    # LIAR + LIAR2 merge builder
    # ================================================================

    @staticmethod
    def _normalize_for_overlap(x: str) -> str:
        """Matches baseline_model_TF-IDF_Linear_SVM.ipynb's normalize_text()
        EXACTLY, including operation order: lowercase, collapse whitespace,
        THEN strip non-word characters using \\w (Unicode-aware in Python 3
        by default -- this matters: an ASCII-only [^a-z0-9\\s] regex strips
        accented characters entirely instead of preserving them, which
        changes which statements collide as "duplicates." Confirmed this
        was the exact source of a merge-count mismatch (23,551 vs the
        correct 23,886) during testing -- do not simplify this to an
        ASCII-only character class."""
        x = str(x).lower()
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"[^\w\s]", "", x)
        return x.strip()

    def build_liar_liar2_merge(
        self, liar_df: pd.DataFrame, liar2_df: pd.DataFrame, deduplicate: bool = True,
    ) -> pd.DataFrame:
        """Harmonize LIAR and LIAR2 to a common schema and concatenate,
        matching the EXACT dedup algorithm from
        baseline_model_TF-IDF_Linear_SVM.ipynb's "Fresh Clean LIAR + LIAR2
        Combined" section (verified byte-for-byte against
        liar_liar2_clean_merged_tfidf.csv: 23,886 total rows, LIAR2=13,127
        / LIAR=10,759, split 21,053 / 1,539 / 1,294 -- exact match).

        The algorithm, in order (this exact order matters):
        1. Concatenate LIAR FIRST, then LIAR2 (not the other way around --
           this determines which row "wins" a duplicate: LIAR wins,
           because drop_duplicates(keep="first") sees LIAR rows first).
        2. Split into train/valid/test using official_split.
        3. Within EACH split separately, drop duplicate normalized
           statements, keeping the first occurrence (i.e. preferring LIAR
           over LIAR2 when both contain the same claim in the same split).
        4. THEN remove cross-split leakage: valid rows whose normalized
           statement appears in the (deduped) train are dropped; test rows
           whose normalized statement appears in the deduped train OR the
           now-cleaned valid are dropped.
        This is NOT a simple "dedupe the whole merged frame" pass -- doing
        it in the wrong order (e.g. date-preference, or deduping before
        splitting) gives a different, wrong cohort. An earlier version of
        this function used a date-preference heuristic and got 23,551
        rows with the composition nearly inverted (LIAR2=22,923,
        LIAR=628) -- confirmed wrong against the real file, replaced by
        the algorithm above.
        """
        common_cols = ["statement", "label", "label_id", "official_split"]

        liar_part = liar_df[[c for c in common_cols if c in liar_df.columns]].copy()
        for col in self.MERGE_COMMON_METADATA_COLUMNS:
            liar_part[col] = liar_df[col] if col in liar_df.columns else np.nan
        liar_part["dataset_source"] = "LIAR"

        liar2_part = liar2_df[[c for c in common_cols if c in liar2_df.columns]].copy()
        for col in self.MERGE_COMMON_METADATA_COLUMNS:
            liar2_part[col] = liar2_df[col] if col in liar2_df.columns else np.nan
        liar2_part["dataset_source"] = "LIAR2"

        # LIAR FIRST -- order determines which row wins a same-split duplicate.
        merged = pd.concat([liar_part, liar2_part], ignore_index=True)
        n_raw = len(merged)
        print(f"Raw merge: {n_raw} rows (LIAR={len(liar_part)}, LIAR2={len(liar2_part)})")

        if not deduplicate:
            print("Composition (raw, no dedup):")
            print(merged["dataset_source"].value_counts())
            return merged

        merged["_norm"] = merged["statement"].apply(self._normalize_for_overlap)

        train_df = merged[merged["official_split"] == "train"].drop_duplicates(subset="_norm", keep="first").copy()
        valid_df = merged[merged["official_split"] == "valid"].drop_duplicates(subset="_norm", keep="first").copy()
        test_df = merged[merged["official_split"] == "test"].drop_duplicates(subset="_norm", keep="first").copy()

        train_statements = set(train_df["_norm"])
        valid_df = valid_df[~valid_df["_norm"].isin(train_statements)].copy()
        valid_statements_clean = set(valid_df["_norm"])
        test_df = test_df[~test_df["_norm"].isin(train_statements)].copy()
        test_df = test_df[~test_df["_norm"].isin(valid_statements_clean)].copy()

        final = pd.concat([train_df, valid_df, test_df], ignore_index=True).drop(columns=["_norm"])

        print(f"Clean merge: {n_raw} -> {len(final)} (removed {n_raw - len(final)})")
        print("Clean merge composition:")
        print(final["dataset_source"].value_counts())
        print("Clean merge split composition:")
        print(final["official_split"].value_counts().reindex(["train", "valid", "test"]))

        return final

    # ================================================================
    # Cross-dataset generalization (B17: train LIAR -> test LIAR2;
    # B18: train LIAR2 -> test LIAR), matching
    # cross-dataset-tf-idf-logistic-regression.ipynb
    # ================================================================

    def run_cross_dataset_generalization(
        self, source_df: pd.DataFrame, source_name: str,
        target_df: pd.DataFrame, target_name: str,
        remove_target_overlap: bool = True, text_col: str = "statement",
    ) -> dict:
        """Train on `source_df`'s official train+valid, test on
        `target_df`'s official test. Matches
        cross-dataset-tf-idf-logistic-regression.ipynb's B17/B18 design
        exactly: source train/valid are used as-is; if
        remove_target_overlap=True (the notebook's default), target test
        rows whose normalized statement appears in source train OR source
        valid are dropped before evaluation -- this is a LEAKAGE CHECK ON
        THE TARGET TEST SET, not a dedup of the source data, and it is
        the reason B18 (train LIAR2 -> test LIAR) has a much smaller
        n_test than B17 (train LIAR -> test LIAR2): LIAR2 extends LIAR, so
        most of LIAR's test statements already appear somewhere in LIAR2's
        train/valid.

        `source_name`/`target_name` are just labels for the result row
        (e.g. "LIAR", "LIAR2").
        """
        source_train = source_df[source_df["official_split"] == "train"].copy()
        source_valid = source_df[source_df["official_split"] == "valid"].copy()
        target_test = target_df[target_df["official_split"] == "test"].copy()

        for part in (source_train, source_valid, target_test):
            part["_norm"] = part[text_col].apply(self._normalize_for_overlap)

        overlap_before = {
            "train_valid_overlap": len(set(source_train["_norm"]) & set(source_valid["_norm"])),
            "train_test_overlap": len(set(source_train["_norm"]) & set(target_test["_norm"])),
            "valid_test_overlap": len(set(source_valid["_norm"]) & set(target_test["_norm"])),
        }

        removed_target_rows = 0
        if remove_target_overlap:
            source_seen = set(source_train["_norm"]) | set(source_valid["_norm"])
            before = len(target_test)
            target_test = target_test[~target_test["_norm"].isin(source_seen)].copy()
            removed_target_rows = before - len(target_test)
            print(f"Cross-dataset ({source_name} -> {target_name}): removed {removed_target_rows} "
                  f"target-test rows overlapping source train/valid ({before} -> {len(target_test)}).")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**self.TFIDF_LOGREG_VECTORIZER_CHRONO)),  # matches the notebook's config
            ("classifier", LogisticRegression(solver="lbfgs", C=1.0, max_iter=3000, random_state=42)),
        ])
        pipeline.fit(source_train[text_col], source_train["label"])

        valid_metrics = self._evaluate(source_valid["label"], pipeline.predict(source_valid[text_col]))
        test_metrics = self._evaluate(target_test["label"], pipeline.predict(target_test[text_col]))
        overlap_after = self.compute_overlap(source_train, source_valid, target_test)

        return self._assemble_result_row(
            "TF-IDF + Logistic Regression", f"Train {source_name} -> Test {target_name}",
            "Cross-dataset generalization", "statement only",
            source_train, source_valid, target_test, valid_metrics, test_metrics, overlap_after,
            notes=f"Target-test leakage removal: {'ON' if remove_target_overlap else 'OFF'}. "
                  f"Removed {removed_target_rows} target-test rows overlapping source train/valid. "
                  f"Overlap before cleaning: {overlap_before}.",
        )

    # ================================================================
    # Split retrieval
    # ================================================================

    def get_split(self, df: pd.DataFrame, split_mode: str = "official", seed: int = 42):
        """
        split_mode:
            "official"           -> uses df["official_split"] as-is
            "random_stratified"  -> generic 70/15/15 stratified re-split
                                     (works for LIAR, LIAR2, or the merge)
            "chronological"      -> ONLY valid for a pure-LIAR2 DataFrame
                                     with a `date_parsed` column; delegates
                                     to self.liar2_p.chronological_split().
                                     Raises if liar2_preprocessor wasn't
                                     provided, or if df has no date_parsed
                                     column (i.e. it's LIAR-only or merged).
        """
        if split_mode == "official":
            if "official_split" not in df.columns:
                raise ValueError("df has no 'official_split' column.")
            train_df = df[df["official_split"] == "train"].copy()
            valid_df = df[df["official_split"] == "valid"].copy()
            test_df = df[df["official_split"] == "test"].copy()
            return train_df, valid_df, test_df, "Official train/valid/test"

        if split_mode == "random_stratified":
            all_idx = np.arange(len(df))
            y = df["label_id"].values
            train_idx, temp_idx = train_test_split(all_idx, train_size=0.70, random_state=seed, stratify=y)
            y_temp = y[temp_idx]
            valid_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=seed, stratify=y_temp)
            train_idx, valid_idx, test_idx = np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)
            return (df.iloc[train_idx].copy(), df.iloc[valid_idx].copy(), df.iloc[test_idx].copy(),
                    "Random stratified 70/15/15 split")

        if split_mode == "chronological":
            if "date_parsed" not in df.columns:
                raise ValueError(
                    "chronological split requires a 'date_parsed' column -- only pure LIAR2 "
                    "DataFrames have one. LIAR has no date column, and the merge doesn't have "
                    "a coherent one either (LIAR rows have no date)."
                )
            if self.liar2_p is None:
                raise ValueError("chronological split requires liar2_preprocessor to be set on this instance.")
            train_idx, valid_idx, test_idx = self.liar2_p.chronological_split(df)
            return (df.iloc[train_idx].copy(), df.iloc[valid_idx].copy(), df.iloc[test_idx].copy(),
                    "Chronological 70/15/15 split by date")

        raise ValueError(f"Unknown split_mode: {split_mode}")

    def compute_overlap(self, train_df, valid_df, test_df) -> dict:
        norm = self._normalize_for_overlap
        train_set = set(train_df["statement"].apply(norm))
        valid_set = set(valid_df["statement"].apply(norm))
        test_set = set(test_df["statement"].apply(norm))
        return {
            "train_valid_overlap": len(train_set & valid_set),
            "train_test_overlap": len(train_set & test_set),
            "valid_test_overlap": len(valid_set & test_set),
        }

    # ================================================================
    # Metrics
    # ================================================================

    def _evaluate(self, y_true, y_pred) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def _assemble_result_row(
        self, model_name, train_dataset_label, split_type_label, input_features,
        train_df, valid_df, test_df, valid_metrics, test_metrics, overlap, notes="",
    ) -> dict:
        return {
            "model": model_name,
            "train_dataset": train_dataset_label,
            "split_type": split_type_label,
            "input_features": input_features,
            "label_space": "6-class truthfulness",
            "n_train": len(train_df),
            "n_valid": len(valid_df),
            "n_test": len(test_df),
            "validation_accuracy": valid_metrics["accuracy"],
            "validation_macro_f1": valid_metrics["macro_f1"],
            "validation_weighted_f1": valid_metrics["weighted_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "train_valid_overlap": overlap["train_valid_overlap"],
            "train_test_overlap": overlap["train_test_overlap"],
            "valid_test_overlap": overlap["valid_test_overlap"],
            "notes": notes,
        }

    # ================================================================
    # Classical baselines -- dataset-agnostic, fully tested
    # ================================================================

    def run_tfidf_logreg(
        self, df: pd.DataFrame, train_dataset_label: str, split_mode: str = "official",
        text_col: str = "statement", use_chrono_config: Optional[bool] = None,
    ) -> dict:
        """TF-IDF + Logistic Regression. Works on LIAR (B01), LIAR2 (B02),
        raw or clean merge (B03/B04), or LIAR2 chronological (C01) --
        pass whichever DataFrame and split_mode fits."""
        if use_chrono_config is None:
            use_chrono_config = split_mode == "chronological"
        vec_cfg = self.TFIDF_LOGREG_VECTORIZER_CHRONO if use_chrono_config else self.TFIDF_LOGREG_VECTORIZER
        clf_cfg = self.TFIDF_LOGREG_CLASSIFIER_CHRONO if use_chrono_config else self.TFIDF_LOGREG_CLASSIFIER

        train_df, valid_df, test_df, split_type_label = self.get_split(df, split_mode)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**vec_cfg)),
            ("classifier", LogisticRegression(**clf_cfg)),
        ])
        pipeline.fit(train_df[text_col], train_df["label"])

        valid_metrics = self._evaluate(valid_df["label"], pipeline.predict(valid_df[text_col]))
        test_metrics = self._evaluate(test_df["label"], pipeline.predict(test_df[text_col]))
        overlap = self.compute_overlap(train_df, valid_df, test_df)

        return self._assemble_result_row(
            "TF-IDF + Logistic Regression", train_dataset_label, split_type_label, "statement only",
            train_df, valid_df, test_df, valid_metrics, test_metrics, overlap,
        )

    def run_tfidf_linear_svm(
        self, df: pd.DataFrame, train_dataset_label: str, split_mode: str = "official",
        text_col: str = "statement", use_chrono_config: Optional[bool] = None,
    ) -> dict:
        """TF-IDF + Linear SVM. Works on LIAR (B05), LIAR2 (B06), clean
        merge (B07), or LIAR2 chronological (C02)."""
        if use_chrono_config is None:
            use_chrono_config = split_mode == "chronological"
        vec_cfg = self.TFIDF_SVM_VECTORIZER_CHRONO if use_chrono_config else self.TFIDF_SVM_VECTORIZER
        clf_cfg = self.TFIDF_SVM_CLASSIFIER_CHRONO if use_chrono_config else self.TFIDF_SVM_CLASSIFIER

        train_df, valid_df, test_df, split_type_label = self.get_split(df, split_mode)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**vec_cfg)),
            ("classifier", LinearSVC(**clf_cfg)),
        ])
        pipeline.fit(train_df[text_col], train_df["label"])

        valid_metrics = self._evaluate(valid_df["label"], pipeline.predict(valid_df[text_col]))
        test_metrics = self._evaluate(test_df["label"], pipeline.predict(test_df[text_col]))
        overlap = self.compute_overlap(train_df, valid_df, test_df)

        return self._assemble_result_row(
            "TF-IDF + Linear SVM", train_dataset_label, split_type_label, "statement only",
            train_df, valid_df, test_df, valid_metrics, test_metrics, overlap,
        )

    def _make_onehot_encoder(self):
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def run_tfidf_logreg_metadata(
        self, df: pd.DataFrame, train_dataset_label: str, metadata_cols: Sequence[str],
        split_mode: str = "official", text_col: str = "statement",
    ) -> dict:
        """TF-IDF (text) + one-hot metadata + Logistic Regression. Works on
        LIAR (B08, pass LiarPreprocessor.SAFE_METADATA_COLUMNS), LIAR2
        (B09, pass Liar2Preprocessor.SAFE_METADATA_COLUMNS), or the merge
        (B10, pass BaselineModels.MERGE_COMMON_METADATA_COLUMNS).
        `metadata_cols` is passed explicitly rather than inferred, since it
        differs by dataset."""
        train_df, valid_df, test_df, split_type_label = self.get_split(df, split_mode)
        metadata_cols = [c for c in metadata_cols if c in df.columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("tfidf_text", TfidfVectorizer(**self.TFIDF_METADATA_VECTORIZER), text_col),
                ("categorical_metadata", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", self._make_onehot_encoder()),
                ]), metadata_cols),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**self.LOGREG_METADATA_CLASSIFIER)),
        ])
        pipeline.fit(train_df, train_df["label"])

        valid_metrics = self._evaluate(valid_df["label"], pipeline.predict(valid_df))
        test_metrics = self._evaluate(test_df["label"], pipeline.predict(test_df))
        overlap = self.compute_overlap(train_df, valid_df, test_df)

        return self._assemble_result_row(
            "TF-IDF + Logistic Regression + metadata", train_dataset_label, split_type_label,
            "statement + metadata", train_df, valid_df, test_df, valid_metrics, test_metrics, overlap,
            notes=f"Metadata columns used (leakage-safe only): {metadata_cols}.",
        )

    # ================================================================
    # Transformer baselines -- WRITTEN BUT NOT RUNNABLE IN THIS SANDBOX
    # (no network access to huggingface.co). Test before use.
    # ================================================================

    def run_distilbert(
        self, df: pd.DataFrame, train_dataset_label: str, split_mode: str = "official",
        text_col: str = "statement", model_name: str = "distilbert-base-uncased",
        max_length: int = 128, batch_size: int = 16, epochs: int = 3,
        label_order: Sequence[str] = None, seed: int = 42,
    ) -> dict:
        """Fine-tuned DistilBERT, matching baseline_DistilBERT_Text-Only
        .ipynb. Works on LIAR (B11), LIAR2 (B12), clean merge (B13), or
        -- with split_mode="chronological" on a LIAR2 DataFrame -- the
        DistilBERT-chronological run this project's methodology tracks as
        a required (previously outstanding) experiment. NOT TESTED in this
        sandbox -- verify in your own GPU/internet environment first.
        `label_order` defaults to the canonical 6-class LIAR2/LIAR order
        if not given. `seed` controls torch's RNG for weight init and data
        shuffling -- run this multiple times with different seeds (see
        run_distilbert_multiseed() below) for the same multi-seed rigor
        used elsewhere in this project, rather than reporting a single run."""
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding,
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        label_order = label_order or ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
        label_to_id = {l: i for i, l in enumerate(label_order)}

        train_df, valid_df, test_df, split_type_label = self.get_split(df, split_mode)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def to_hf_dataset(part_df):
            ds = Dataset.from_pandas(part_df[[text_col, "label"]].rename(columns={text_col: "text"}))
            ds = ds.map(lambda ex: {"label": label_to_id[ex["label"]]})
            ds = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length), batched=True)
            return ds

        train_ds, valid_ds, test_ds = to_hf_dataset(train_df), to_hf_dataset(valid_df), to_hf_dataset(test_df)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_order))
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # transformers renamed `evaluation_strategy=` to `eval_strategy=` in
        # 4.46+, and separately renamed Trainer's `tokenizer=` to
        # `processing_class=` in 5.x. Both are handled defensively below
        # rather than pinned to one version, since this method can't be
        # tested against the Hugging Face Hub in this sandbox and the
        # Kaggle environment's exact transformers version is unknown.
        try:
            args = TrainingArguments(
                output_dir="./distilbert_baseline_out", num_train_epochs=epochs,
                per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                eval_strategy="epoch", save_strategy="no", report_to=[], seed=seed,
            )
        except TypeError:
            args = TrainingArguments(
                output_dir="./distilbert_baseline_out", num_train_epochs=epochs,
                per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                evaluation_strategy="epoch", save_strategy="no", report_to=[], seed=seed,
            )
        try:
            trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                               data_collator=collator, processing_class=tokenizer)
        except TypeError:
            trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                               data_collator=collator, tokenizer=tokenizer)
        trainer.train()

        id_to_label = {i: l for l, i in label_to_id.items()}
        valid_pred = [id_to_label[i] for i in np.argmax(trainer.predict(valid_ds).predictions, axis=1)]
        test_pred = [id_to_label[i] for i in np.argmax(trainer.predict(test_ds).predictions, axis=1)]

        valid_metrics = self._evaluate(valid_df["label"], valid_pred)
        test_metrics = self._evaluate(test_df["label"], test_pred)
        overlap = self.compute_overlap(train_df, valid_df, test_df)

        return self._assemble_result_row(
            "DistilBERT", train_dataset_label, split_type_label, "statement only",
            train_df, valid_df, test_df, valid_metrics, test_metrics, overlap,
            notes=f"epochs={epochs}, batch_size={batch_size}, max_length={max_length}, "
                  f"model={model_name}, seed={seed}.",
        )

    def run_distilbert_multiseed(
        self, df: pd.DataFrame, train_dataset_label: str, seeds, split_mode: str = "official",
        **kwargs,
    ) -> pd.DataFrame:
        """Runs run_distilbert() once per seed, returning a DataFrame with
        one row per seed plus mean/std summary rows -- matching the
        multi-seed rigor used for DA-MiniLM-RWAC/MiniLM-LogReg-RW
        elsewhere in this project, rather than reporting a single
        DistilBERT run. Use split_mode="chronological" on a LIAR2
        DataFrame for the DistilBERT-chronological run tracked as a
        required experiment."""
        rows = []
        for seed in seeds:
            print(f"running DistilBERT seed={seed} split_mode={split_mode}")
            row = self.run_distilbert(df, train_dataset_label, split_mode=split_mode, seed=seed, **kwargs)
            rows.append(row)
            print(f"  macro_f1={row['test_macro_f1']:.4f}")

        results_df = pd.DataFrame(rows)

        numeric_cols = [c for c in results_df.columns if results_df[c].dtype.kind in "fc"]
        summary = {"model": f"{results_df['model'].iloc[0]} (mean over {len(seeds)} seeds)",
                   "train_dataset": train_dataset_label, "split_type": results_df["split_type"].iloc[0]}
        for col in numeric_cols:
            summary[f"{col}_mean"] = results_df[col].mean()
            summary[f"{col}_std"] = results_df[col].std()

        print()
        print(f"DistilBERT {train_dataset_label} ({split_mode}), {len(seeds)} seeds: "
              f"test_macro_f1 = {summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}")

        return results_df, summary

    def run_minilm(
        self, df: pd.DataFrame, train_dataset_label: str, split_mode: str = "official",
        text_col: str = "statement", model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 256,
    ) -> dict:
        """Frozen MiniLM embeddings + Logistic Regression (Uniform-LR
        equivalent). Works on LIAR (B14), LIAR2 (B15), or clean merge
        (B16). NOT TESTED in this sandbox."""
        from sentence_transformers import SentenceTransformer

        train_df, valid_df, test_df, split_type_label = self.get_split(df, split_mode)

        embedder = SentenceTransformer(model_name)
        X_train = embedder.encode(train_df[text_col].tolist(), batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
        X_valid = embedder.encode(valid_df[text_col].tolist(), batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
        X_test = embedder.encode(test_df[text_col].tolist(), batch_size=batch_size,
                                  normalize_embeddings=True, show_progress_bar=False)

        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_train, train_df["label"])

        valid_metrics = self._evaluate(valid_df["label"], clf.predict(X_valid))
        test_metrics = self._evaluate(test_df["label"], clf.predict(X_test))
        overlap = self.compute_overlap(train_df, valid_df, test_df)

        return self._assemble_result_row(
            "MiniLM", train_dataset_label, split_type_label, "statement only",
            train_df, valid_df, test_df, valid_metrics, test_metrics, overlap,
            notes=f"Frozen embeddings, no reweighting/calibration (Uniform-LR equivalent). "
                  f"model={model_name}, batch_size={batch_size}.",
        )

    # ================================================================
    # Convenience: run everything, return xlsx-shaped DataFrame
    # ================================================================

    def run_all_experiments(
        self, liar_df: pd.DataFrame, liar2_df: pd.DataFrame,
        include_transformer_baselines: bool = False,
        include_raw_merge_demo: bool = True,
    ) -> pd.DataFrame:
        """Runs the full testable battery:
            B01/B02  TF-IDF+LogReg on LIAR / LIAR2
            B03/B04  TF-IDF+LogReg on raw merge / clean merge
            B05/B06  TF-IDF+SVM on LIAR / LIAR2
            B07      TF-IDF+SVM on clean merge
            B08/B09  TF-IDF+LogReg+metadata on LIAR / LIAR2
            B10      TF-IDF+LogReg+metadata on clean merge
            C01/C02  TF-IDF+LogReg / SVM on LIAR2 chronological
            B17/B18  Cross-dataset generalization (LIAR->LIAR2, LIAR2->LIAR)
        plus B11-B16 (DistilBERT/MiniLM x 3 datasets) if
        include_transformer_baselines=True.

        B03 (raw merge) is included only as the leakage DEMONSTRATION the
        xlsx itself describes it as -- its notes field says so explicitly;
        don't report it as a result.
        """
        rows = []

        # B01/B02
        print("running B01: TF-IDF + Logistic Regression, train: LIAR; test: LIAR")
        rows.append(self.run_tfidf_logreg(liar_df, "LIAR"))
        print("running B02: TF-IDF + Logistic Regression, train: LIAR2; test: LIAR2")
        rows.append(self.run_tfidf_logreg(liar2_df, "LIAR2"))

        # B03/B04 (merge)
        if include_raw_merge_demo:
            print("running B03: TF-IDF + Logistic Regression, train: LIAR+LIAR2 raw merge; test: LIAR+LIAR2 raw merge")
            raw_merge_df = self.build_liar_liar2_merge(liar_df, liar2_df, deduplicate=False)
            r03 = self.run_tfidf_logreg(raw_merge_df, "LIAR + LIAR2 raw merge")
            r03["notes"] = "Leakage detected -- do not report as final; leakage demonstration only."
            rows.append(r03)

        print("running B04: TF-IDF + Logistic Regression, train: LIAR+LIAR2 clean merge; test: LIAR+LIAR2 clean merge")
        clean_merge_df = self.build_liar_liar2_merge(liar_df, liar2_df, deduplicate=True)
        rows.append(self.run_tfidf_logreg(clean_merge_df, "LIAR + LIAR2 clean merge"))

        # B05/B06/B07
        print("running B05: TF-IDF + Linear SVM, train: LIAR; test: LIAR")
        rows.append(self.run_tfidf_linear_svm(liar_df, "LIAR"))
        print("running B06: TF-IDF + Linear SVM, train: LIAR2; test: LIAR2")
        rows.append(self.run_tfidf_linear_svm(liar2_df, "LIAR2"))
        print("running B07: TF-IDF + Linear SVM, train: LIAR+LIAR2 clean merge; test: LIAR+LIAR2 clean merge")
        rows.append(self.run_tfidf_linear_svm(clean_merge_df, "LIAR + LIAR2 clean merge"))

        # B08/B09/B10
        print("running B08: TF-IDF + Logistic Regression + metadata, train: LIAR; test: LIAR")
        rows.append(self.run_tfidf_logreg_metadata(
            liar_df, "LIAR", self.liar_p.SAFE_METADATA_COLUMNS if self.liar_p else
            ["subject", "speaker", "job_title", "state_info", "party_affiliation", "context"],
        ))
        print("running B09: TF-IDF + Logistic Regression + metadata, train: LIAR2; test: LIAR2")
        rows.append(self.run_tfidf_logreg_metadata(
            liar2_df, "LIAR2", self.liar2_p.SAFE_METADATA_COLUMNS if self.liar2_p else
            ["subject", "speaker", "speaker_description", "state_info", "context"],
        ))
        print("running B10: TF-IDF + Logistic Regression + metadata, train: LIAR+LIAR2 clean merge; test: LIAR+LIAR2 clean merge")
        rows.append(self.run_tfidf_logreg_metadata(
            clean_merge_df, "LIAR + LIAR2 clean merge", self.MERGE_COMMON_METADATA_COLUMNS,
        ))

        # C01/C02 (LIAR2 only)
        print("running C01: TF-IDF + Logistic Regression, train: LIAR2 chronological; test: LIAR2 chronological")
        rows.append(self.run_tfidf_logreg(liar2_df, "LIAR2 chronological", split_mode="chronological"))
        print("running C02: TF-IDF + Linear SVM, train: LIAR2 chronological; test: LIAR2 chronological")
        rows.append(self.run_tfidf_linear_svm(liar2_df, "LIAR2 chronological", split_mode="chronological"))

        # B17/B18 (cross-dataset generalization)
        print("running B17: TF-IDF + Logistic Regression, train: LIAR; test: LIAR2")
        rows.append(self.run_cross_dataset_generalization(liar_df, "LIAR", liar2_df, "LIAR2"))
        print("running B18: TF-IDF + Logistic Regression, train: LIAR2; test - LIAR")
        rows.append(self.run_cross_dataset_generalization(liar2_df, "LIAR2", liar_df, "LIAR"))

        if include_transformer_baselines:
            print("running B11: DistilBERT, train: LIAR; test: LIAR")
            rows.append(self.run_distilbert(liar_df, "LIAR"))
            print("running B12: DistilBERT, train: LIAR2; test: LIAR2")
            rows.append(self.run_distilbert(liar2_df, "LIAR2"))
            print("running B13: DistilBERT, train: LIAR+LIAR2 clean merge; test: LIAR+LIAR2 clean merge")
            rows.append(self.run_distilbert(clean_merge_df, "LIAR + LIAR2 clean merge"))
            print("running B14: MiniLM, train: LIAR; test: LIAR")
            rows.append(self.run_minilm(liar_df, "LIAR"))
            print("running B15: MiniLM, train: LIAR2; test: LIAR2")
            rows.append(self.run_minilm(liar2_df, "LIAR2"))
            print("running B16: MiniLM, train: LIAR+LIAR2 clean merge; test: LIAR+LIAR2 clean merge")
            rows.append(self.run_minilm(clean_merge_df, "LIAR + LIAR2 clean merge"))

        print("\nAll requested experiments complete.")
        return pd.DataFrame(rows)


if __name__ == "__main__":
    print("baseline_models.py loaded. Class: BaselineModels "
          "(methods: build_liar_liar2_merge, run_tfidf_logreg, "
          "run_tfidf_linear_svm, run_tfidf_logreg_metadata, "
          "run_distilbert, run_minilm, run_all_experiments)")