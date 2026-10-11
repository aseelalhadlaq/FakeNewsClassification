"""
drift_analysis.py

Implements D01-D05 from d01-d05-drift-experiments.ipynb: label, vocabulary,
metadata, embedding, and domain-classifier drift diagnostics comparing
LIAR2's chronological train split against its chronological test split.

Uses Liar2Preprocessor for all data loading/cleaning -- this file only
computes drift statistics on top of what that class produces. To match
the original notebook's cohort exactly, the DataFrame passed in must be
deduplicated AND chronologically re-split first:

    from liar2_preprocessor import Liar2Preprocessor
    p = Liar2Preprocessor()
    df = p.run_pipeline(train_path, valid_path, test_path)
    audit = p.audit_duplicate_statements(df)
    df = p.drop_duplicate_statements(df, policy="keep_first", audit_df=audit)
    train_idx, valid_idx, test_idx = p.chronological_split(df)
    train_df, valid_df, test_df = df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]

This exact sequence (dedup THEN chronological split) reproduces the
original notebook's reported cohort sizes (16,046 / 3,438 / 3,439) --
confirmed in an earlier step of this project. Skipping the dedup step
before splitting will NOT match the reported sample sizes.

One schema note: the original notebook's METADATA_COLUMNS list includes
`party_affiliation`, which does not exist in LIAR2's actual schema (it is
a LIAR-only field) -- confirmed against the real uploaded LIAR2 CSVs
(columns: id, label, statement, date, subject, speaker,
speaker_description, state_info, true_counts, ..., context,
justification). `run_metadata_drift()` here only uses columns that
genuinely exist on the LIAR2 DataFrame it's given (subject, speaker,
speaker_description, state_info, context, and optionally official_split
as a stand-in for the notebook's "original_split"), silently skipping
anything absent -- matching the original notebook's own
`available_metadata_cols` filtering behavior, not erroring on it.

D04 (MiniLM embedding shift) requires downloading
`microsoft/MiniLM-L12-H384-uncased` from the Hugging Face Hub, which this
sandbox's network cannot reach. It is written to match the notebook
exactly but NOT executed here -- run and verify it yourself before
trusting its output. D01, D02, D03, and D05 require no network access and
are tested end-to-end against real data in this project.

Usage
-----
    from drift_analysis import DriftAnalysis

    da = DriftAnalysis()
    d01 = da.run_label_drift(train_df, test_df)
    d02 = da.run_vocabulary_drift(train_df, test_df)
    d03 = da.run_metadata_drift(train_df, test_df)
    d05 = da.run_domain_classifier(train_df, test_df)

    # everything at once, plus a comparison against the originally
    # reported values:
    summary_df = da.run_all_drift_experiments(train_df, test_df, include_embedding_shift=False)
    da.compare_to_reported(summary_df)
"""

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.decomposition import PCA


# Originally reported values from drift_shifting_experiment_results_liar2.xlsx
# ("Drift Metrics" sheet), for compare_to_reported().
REPORTED_VALUES = {
    "D01": {"js_distance": 0.396195, "total_variation": 0.428790},
    "D02": {"js_distance": 0.419836, "total_variation": 0.376209},
    "D03": {"speaker_js": 0.645526, "context_js": 0.597155},
    "D04": {"mmd_rbf": 0.025154, "centroid_cosine_distance": 0.003976},
    "D05": {"domain_auc": 0.908728, "domain_accuracy": 0.886418},
}


class DriftAnalysis:

    def __init__(
        self,
        label_order: Sequence[str] = None,
        random_state: int = 42,
        vocab_max_features: int = 30000,
        vocab_min_df: int = 2,
        top_shifted_terms: int = 40,
        top_metadata_categories: int = 50,
        embedding_model_name: str = "microsoft/MiniLM-L12-H384-uncased",
        embedding_sample_size_per_split: int = 2000,
        embedding_batch_size: int = 32,
        embedding_max_length: int = 128,
        domain_test_size: float = 0.30,
    ):
        self.label_order = label_order or [
            "pants-fire", "false", "barely-true", "half-true", "mostly-true", "true",
        ]
        self.random_state = random_state
        self.vocab_max_features = vocab_max_features
        self.vocab_min_df = vocab_min_df
        self.top_shifted_terms = top_shifted_terms
        self.top_metadata_categories = top_metadata_categories
        self.embedding_model_name = embedding_model_name
        self.embedding_sample_size_per_split = embedding_sample_size_per_split
        self.embedding_batch_size = embedding_batch_size
        self.embedding_max_length = embedding_max_length
        self.domain_test_size = domain_test_size

    # ================================================================
    # Shared utilities (match the notebook exactly)
    # ================================================================

    @staticmethod
    def normalize_text(text) -> str:
        text = str(text).lower().strip()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def safe_distribution(series: pd.Series, categories: Sequence[str]) -> np.ndarray:
        counts = series.value_counts()
        values = np.array([counts.get(c, 0) for c in categories], dtype=float)
        if values.sum() == 0:
            return np.ones(len(categories)) / len(categories)
        return values / values.sum()

    @staticmethod
    def js_distance(p, q) -> float:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p / p.sum()
        q = q / q.sum()
        return float(jensenshannon(p, q, base=2))

    @staticmethod
    def total_variation_distance(p, q) -> float:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p / p.sum()
        q = q / q.sum()
        return float(0.5 * np.abs(p - q).sum())

    @staticmethod
    def interpret_js_distance(value: float) -> str:
        if value < 0.05:
            return "very weak drift"
        elif value < 0.10:
            return "weak drift"
        elif value < 0.20:
            return "moderate drift"
        return "strong drift"

    @staticmethod
    def interpret_domain_auc(auc: float) -> str:
        if auc < 0.60:
            return "weak domain separability"
        elif auc < 0.75:
            return "moderate domain separability"
        elif auc < 0.90:
            return "strong domain separability"
        return "very strong domain separability"

    # ================================================================
    # D01: Label drift
    # ================================================================

    def run_label_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        train_p = self.safe_distribution(train_df["label"], self.label_order)
        test_p = self.safe_distribution(test_df["label"], self.label_order)

        label_dist_df = pd.DataFrame({
            "label": self.label_order,
            "train_count": [train_df["label"].value_counts().get(l, 0) for l in self.label_order],
            "test_count": [test_df["label"].value_counts().get(l, 0) for l in self.label_order],
            "train_proportion": train_p,
            "test_proportion": test_p,
            "abs_train_test_difference": np.abs(train_p - test_p),
        })

        contingency = np.vstack([
            [train_df["label"].value_counts().get(l, 0) for l in self.label_order],
            [test_df["label"].value_counts().get(l, 0) for l in self.label_order],
        ])
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        js = self.js_distance(train_p, test_p)
        summary = {
            "experiment_id": "D01",
            "analysis": "Label drift",
            "comparison": "chronological train vs test",
            "train_n": int(len(train_df)),
            "test_n": int(len(test_df)),
            "js_distance_train_test": js,
            "total_variation_train_test": self.total_variation_distance(train_p, test_p),
            "chi_square_train_test": float(chi2),
            "chi_square_p_value_train_test": float(p_value),
            "interpretation": self.interpret_js_distance(js),
        }
        return {"summary": summary, "label_distribution": label_dist_df}

    # ================================================================
    # D02: Vocabulary drift
    # ================================================================

    def run_vocabulary_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str = "statement") -> dict:
        train_texts = train_df[text_col].map(self.normalize_text).tolist()
        test_texts = test_df[text_col].map(self.normalize_text).tolist()

        vectorizer = CountVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=self.vocab_max_features, min_df=self.vocab_min_df,
        )
        X_all = vectorizer.fit_transform(train_texts + test_texts)
        X_train = X_all[:len(train_texts)]
        X_test = X_all[len(train_texts):]

        train_counts = np.asarray(X_train.sum(axis=0)).ravel().astype(float)
        test_counts = np.asarray(X_test.sum(axis=0)).ravel().astype(float)

        # Add-one smoothing
        train_probs = (train_counts + 1.0)
        test_probs = (test_counts + 1.0)
        train_probs = train_probs / train_probs.sum()
        test_probs = test_probs / test_probs.sum()

        vocab = np.array(vectorizer.get_feature_names_out())
        difference = test_probs - train_probs

        vocab_df = pd.DataFrame({
            "term": vocab, "train_probability": train_probs, "test_probability": test_probs,
            "test_minus_train": difference, "absolute_difference": np.abs(difference),
        })
        terms_more_in_train = vocab_df.sort_values("test_minus_train", ascending=True).head(self.top_shifted_terms)
        terms_more_in_test = vocab_df.sort_values("test_minus_train", ascending=False).head(self.top_shifted_terms)

        js = self.js_distance(train_probs, test_probs)
        summary = {
            "experiment_id": "D02",
            "analysis": "Vocabulary drift",
            "comparison": "chronological train vs test",
            "train_n": int(len(train_df)),
            "test_n": int(len(test_df)),
            "n_vocabulary_terms": int(len(vocab)),
            "js_distance_train_test": js,
            "total_variation_train_test": self.total_variation_distance(train_probs, test_probs),
            "interpretation": self.interpret_js_distance(js),
        }
        return {
            "summary": summary, "vocab_df": vocab_df,
            "terms_more_in_train": terms_more_in_train, "terms_more_in_test": terms_more_in_test,
        }

    # ================================================================
    # D03: Metadata drift
    # ================================================================

    def _prepare_metadata_categories(self, train_series: pd.Series, test_series: pd.Series, top_k: int):
        combined = pd.concat([train_series, test_series], axis=0).astype(str).fillna("MISSING").str.strip()
        combined = combined.replace({"": "MISSING", "nan": "MISSING", "None": "MISSING", "none": "MISSING"})
        top_categories = combined.value_counts().head(top_k).index.tolist()

        train_clean = train_series.astype(str).fillna("MISSING").str.strip()
        test_clean = test_series.astype(str).fillna("MISSING").str.strip()
        train_clean = train_clean.where(train_clean.isin(top_categories), "OTHER")
        test_clean = test_clean.where(test_clean.isin(top_categories), "OTHER")

        return train_clean, test_clean, top_categories + ["OTHER"]

    def run_metadata_drift(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
        metadata_columns: Optional[Sequence[str]] = None,
    ) -> dict:
        """Defaults to LIAR2's actual metadata columns (subject, speaker,
        speaker_description, state_info, context, official_split) --
        NOT the original notebook's list verbatim, which included
        `party_affiliation` (a LIAR-only field absent from real LIAR2) and
        `original_split` (renamed here to `official_split`, matching
        Liar2Preprocessor's column name). Any column not present on the
        given DataFrames is silently skipped, matching the notebook's own
        `available_metadata_cols` filtering behavior."""
        metadata_columns = metadata_columns or [
            "subject", "speaker", "speaker_description", "state_info", "context", "official_split",
        ]
        available_cols = [c for c in metadata_columns if c in train_df.columns and c in test_df.columns]
        if not available_cols:
            raise ValueError("No metadata columns found for D03.")

        summary_rows = []
        shifted_category_rows = []

        for col in available_cols:
            train_col, test_col, categories = self._prepare_metadata_categories(
                train_df[col], test_df[col], top_k=self.top_metadata_categories,
            )
            train_p = self.safe_distribution(train_col, categories)
            test_p = self.safe_distribution(test_col, categories)
            col_js = self.js_distance(train_p, test_p)
            col_tvd = self.total_variation_distance(train_p, test_p)

            contingency = np.vstack([
                [train_col.value_counts().get(c, 0) for c in categories],
                [test_col.value_counts().get(c, 0) for c in categories],
            ])
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
            except Exception:
                chi2, p_value = np.nan, np.nan

            summary_rows.append({
                "experiment_id": "D03", "analysis": "Metadata drift",
                "comparison": "chronological train vs test", "metadata_column": col,
                "n_categories_after_topk": len(categories),
                "js_distance_train_test": col_js, "total_variation_train_test": col_tvd,
                "chi_square_train_test": chi2, "chi_square_p_value_train_test": p_value,
                "interpretation": self.interpret_js_distance(col_js),
            })

            shifted_category_rows.append(pd.DataFrame({
                "metadata_column": col, "category": categories,
                "train_proportion": train_p, "test_proportion": test_p,
                "test_minus_train": test_p - train_p, "absolute_difference": np.abs(test_p - train_p),
            }).sort_values("absolute_difference", ascending=False))

        summary_df = pd.DataFrame(summary_rows).sort_values("js_distance_train_test", ascending=False)
        shifted_categories_df = pd.concat(shifted_category_rows, ignore_index=True)

        return {"summary_df": summary_df, "shifted_categories_df": shifted_categories_df}

    # ================================================================
    # D04: MiniLM embedding shift -- WRITTEN BUT NOT RUNNABLE IN THIS
    # SANDBOX (no network access to huggingface.co). Test before use.
    # ================================================================

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        import torch
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _encode_texts_minilm(self, texts, tokenizer, model, device):
        import torch
        model.eval()
        all_embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), self.embedding_batch_size):
                batch_texts = texts[start:start + self.embedding_batch_size]
                enc = tokenizer(batch_texts, padding=True, truncation=True,
                                 max_length=self.embedding_max_length, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                outputs = model(**enc)
                emb = self._mean_pooling(outputs.last_hidden_state, enc["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)

    def _stratified_sample_for_embeddings(self, df_part: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        if len(df_part) <= sample_size:
            return df_part.copy()
        sampled_parts = []
        for label, group in df_part.groupby("label"):
            label_frac = len(group) / len(df_part)
            label_n = max(1, int(round(sample_size * label_frac)))
            sampled_parts.append(group.sample(n=min(label_n, len(group)), random_state=self.random_state))
        sampled = pd.concat(sampled_parts, ignore_index=True)
        if len(sampled) > sample_size:
            sampled = sampled.sample(n=sample_size, random_state=self.random_state)
        return sampled.reset_index(drop=True)

    def _compute_mmd_rbf(self, X: np.ndarray, Y: np.ndarray, max_mmd_samples: int = 1000):
        if len(X) > max_mmd_samples:
            rng = np.random.default_rng(self.random_state)
            X = X[rng.choice(len(X), max_mmd_samples, replace=False)]
        if len(Y) > max_mmd_samples:
            rng = np.random.default_rng(self.random_state)
            Y = Y[rng.choice(len(Y), max_mmd_samples, replace=False)]

        combined = np.vstack([X, Y])
        pairwise_sq = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1)
        median_sq = np.median(pairwise_sq[pairwise_sq > 0])
        gamma = 1.0 if (median_sq <= 0 or np.isnan(median_sq)) else 1.0 / median_sq

        Kxx = rbf_kernel(X, X, gamma=gamma)
        Kyy = rbf_kernel(Y, Y, gamma=gamma)
        Kxy = rbf_kernel(X, Y, gamma=gamma)
        mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        return float(mmd), float(gamma)

    def run_embedding_shift(self, train_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str = "statement") -> dict:
        """NOT TESTED in this sandbox -- requires downloading
        microsoft/MiniLM-L12-H384-uncased from the Hugging Face Hub, which
        this environment's network cannot reach. Verify in your own
        internet-enabled environment before trusting the output."""
        import torch
        from transformers import AutoTokenizer, AutoModel

        train_sample = self._stratified_sample_for_embeddings(train_df, self.embedding_sample_size_per_split)
        test_sample = self._stratified_sample_for_embeddings(test_df, self.embedding_sample_size_per_split)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        model = AutoModel.from_pretrained(self.embedding_model_name).to(device)

        train_emb = self._encode_texts_minilm(train_sample[text_col].tolist(), tokenizer, model, device)
        test_emb = self._encode_texts_minilm(test_sample[text_col].tolist(), tokenizer, model, device)

        train_centroid = train_emb.mean(axis=0, keepdims=True)
        test_centroid = test_emb.mean(axis=0, keepdims=True)
        centroid_cosine_similarity = cosine_similarity(train_centroid, test_centroid)[0, 0]
        centroid_cosine_distance = 1.0 - centroid_cosine_similarity

        mmd_rbf, mmd_gamma = self._compute_mmd_rbf(train_emb, test_emb)

        summary = {
            "experiment_id": "D04", "analysis": "MiniLM embedding shift",
            "comparison": "chronological train vs test", "embedding_model": self.embedding_model_name,
            "train_sample_size": int(len(train_sample)), "test_sample_size": int(len(test_sample)),
            "embedding_dim": int(train_emb.shape[1]),
            "centroid_cosine_similarity": float(centroid_cosine_similarity),
            "centroid_cosine_distance": float(centroid_cosine_distance),
            "mmd_rbf": float(mmd_rbf), "mmd_rbf_gamma": float(mmd_gamma),
            "interpretation": "larger centroid distance and MMD indicate stronger semantic embedding shift",
        }

        all_emb = np.vstack([train_emb, test_emb])
        split_labels = np.array(["train"] * len(train_emb) + ["test"] * len(test_emb))
        pca = PCA(n_components=2, random_state=self.random_state)
        pca_2d = pca.fit_transform(all_emb)
        pca_df = pd.DataFrame({"pc1": pca_2d[:, 0], "pc2": pca_2d[:, 1], "split": split_labels})

        return {"summary": summary, "pca_df": pca_df}

    # ================================================================
    # D05: Domain classifier drift
    # ================================================================

    def run_domain_classifier(self, train_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str = "statement") -> dict:
        old_df = train_df[[text_col]].rename(columns={text_col: "text"}).copy()
        new_df = test_df[[text_col]].rename(columns={text_col: "text"}).copy()
        old_df["domain"] = 0
        new_df["domain"] = 1

        domain_df = pd.concat([old_df, new_df], ignore_index=True)
        domain_df = domain_df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        X_train, X_eval, y_train, y_eval = train_test_split(
            domain_df["text"], domain_df["domain"], test_size=self.domain_test_size,
            random_state=self.random_state, stratify=domain_df["domain"],
        )

        domain_clf = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", stop_words="english",
                ngram_range=(1, 2), max_features=50000, min_df=2, max_df=0.95, sublinear_tf=True,
            )),
            # solver="liblinear" is safe here (unlike the multiclass case
            # elsewhere in this project) -- this is a binary classification
            # problem (old domain vs new domain), and liblinear's binary
            # path was never affected by the multiclass-OvR removal that
            # broke the 6-class baselines on current scikit-learn.
            ("clf", LogisticRegression(solver="liblinear", C=1.0, max_iter=3000, random_state=self.random_state)),
        ])
        domain_clf.fit(X_train, y_train)

        y_pred = domain_clf.predict(X_eval)
        y_score = domain_clf.predict_proba(X_eval)[:, 1]

        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred)
        auc = roc_auc_score(y_eval, y_score)

        summary = {
            "experiment_id": "D05", "analysis": "Domain classifier drift",
            "comparison": "chronological train vs test",
            "domain_0": "old chronological train", "domain_1": "new chronological test",
            "n_total": int(len(domain_df)), "n_eval": int(len(X_eval)),
            "domain_accuracy": float(acc), "domain_f1": float(f1), "domain_auc": float(auc),
            "interpretation": self.interpret_domain_auc(auc),
        }

        report_str = classification_report(
            y_eval, y_pred, target_names=["old_train_domain", "new_test_domain"], digits=4,
        )

        vectorizer = domain_clf.named_steps["tfidf"]
        classifier = domain_clf.named_steps["clf"]
        feature_names = np.array(vectorizer.get_feature_names_out())
        coefs = classifier.coef_[0]
        top_new_idx = np.argsort(coefs)[-40:][::-1]
        top_old_idx = np.argsort(coefs)[:40]
        top_domain_terms = pd.concat([
            pd.DataFrame({"term": feature_names[top_new_idx], "coefficient": coefs[top_new_idx],
                          "more_associated_with": "new_test_domain"}),
            pd.DataFrame({"term": feature_names[top_old_idx], "coefficient": coefs[top_old_idx],
                          "more_associated_with": "old_train_domain"}),
        ], ignore_index=True)

        return {"summary": summary, "classification_report": report_str, "top_domain_terms": top_domain_terms}

    # ================================================================
    # Convenience: run everything, plus comparison against reported values
    # ================================================================

    def run_all_drift_experiments(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
        metadata_columns: Optional[Sequence[str]] = None,
        include_embedding_shift: bool = False,
    ) -> pd.DataFrame:
        """Runs D01, D02, D03, D05 (and D04 if include_embedding_shift=True
        -- requires Hugging Face Hub access). Returns a DataFrame with one
        row per D0x summary (D03 contributes one row per metadata column)."""
        rows = []

        print("running D01: Label drift")
        rows.append(self.run_label_drift(train_df, test_df)["summary"])

        print("running D02: Vocabulary drift")
        rows.append(self.run_vocabulary_drift(train_df, test_df)["summary"])

        print("running D03: Metadata drift")
        d03 = self.run_metadata_drift(train_df, test_df, metadata_columns=metadata_columns)
        rows.extend(d03["summary_df"].to_dict("records"))

        if include_embedding_shift:
            print("running D04: MiniLM embedding shift")
            rows.append(self.run_embedding_shift(train_df, test_df)["summary"])

        print("running D05: Domain classifier drift")
        rows.append(self.run_domain_classifier(train_df, test_df)["summary"])

        print("\nAll drift experiments complete.")
        return pd.DataFrame(rows)

    def compare_to_reported(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Compares this run's D01/D02/D05 metrics (and D03's speaker/
        context JS if present) against REPORTED_VALUES, the figures from
        drift_shifting_experiment_results_liar2.xlsx's "Drift Metrics"
        sheet. Returns a small comparison table; also prints it."""
        rows = []

        d01 = summary_df[summary_df["experiment_id"] == "D01"]
        if len(d01):
            r = d01.iloc[0]
            rows.append(("D01 JS distance", r["js_distance_train_test"], REPORTED_VALUES["D01"]["js_distance"]))
            rows.append(("D01 total variation", r["total_variation_train_test"], REPORTED_VALUES["D01"]["total_variation"]))

        d02 = summary_df[summary_df["experiment_id"] == "D02"]
        if len(d02):
            r = d02.iloc[0]
            rows.append(("D02 JS distance", r["js_distance_train_test"], REPORTED_VALUES["D02"]["js_distance"]))
            rows.append(("D02 total variation", r["total_variation_train_test"], REPORTED_VALUES["D02"]["total_variation"]))

        d03 = summary_df[summary_df["experiment_id"] == "D03"]
        if len(d03) and "metadata_column" in d03.columns:
            speaker_row = d03[d03["metadata_column"] == "speaker"]
            if len(speaker_row):
                rows.append(("D03 speaker JS", speaker_row.iloc[0]["js_distance_train_test"],
                             REPORTED_VALUES["D03"]["speaker_js"]))
            context_row = d03[d03["metadata_column"] == "context"]
            if len(context_row):
                rows.append(("D03 context JS", context_row.iloc[0]["js_distance_train_test"],
                             REPORTED_VALUES["D03"]["context_js"]))

        d04 = summary_df[summary_df["experiment_id"] == "D04"]
        if len(d04):
            r = d04.iloc[0]
            rows.append(("D04 MMD (RBF)", r["mmd_rbf"], REPORTED_VALUES["D04"]["mmd_rbf"]))
            rows.append(("D04 centroid cosine distance", r["centroid_cosine_distance"],
                         REPORTED_VALUES["D04"]["centroid_cosine_distance"]))

        d05 = summary_df[summary_df["experiment_id"] == "D05"]
        if len(d05):
            r = d05.iloc[0]
            rows.append(("D05 domain AUC", r["domain_auc"], REPORTED_VALUES["D05"]["domain_auc"]))
            rows.append(("D05 domain accuracy", r["domain_accuracy"], REPORTED_VALUES["D05"]["domain_accuracy"]))

        comparison_df = pd.DataFrame(rows, columns=["metric", "this_run", "originally_reported"])
        comparison_df["abs_difference"] = (comparison_df["this_run"] - comparison_df["originally_reported"]).abs()

        print(comparison_df.to_string(index=False))
        return comparison_df


if __name__ == "__main__":
    print("drift_analysis.py loaded. Class: DriftAnalysis "
          "(methods: run_label_drift, run_vocabulary_drift, run_metadata_drift, "
          "run_embedding_shift, run_domain_classifier, run_all_drift_experiments, "
          "compare_to_reported)")