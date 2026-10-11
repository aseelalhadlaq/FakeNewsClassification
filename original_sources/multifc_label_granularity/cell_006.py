# ============================================================
# 1a. MultiFCPreprocessor  (replaces Liar2Preprocessor)
# ============================================================

# 3-class harmonised taxonomy — identical to the collapsed LIAR2 taxonomy, so the
# two datasets' 3-class numbers are directly comparable.
MULTIFC_LABEL_ORDER = ["false", "mixed", "true"]
MULTIFC_LABEL_TO_ID = {l: i for i, l in enumerate(MULTIFC_LABEL_ORDER)}
MULTIFC_ID_TO_LABEL = {i: l for l, i in MULTIFC_LABEL_TO_ID.items()}


def _mfc_norm_label(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    x = str(x).strip().lower().replace("_", "-")
    return re.sub(r"\s+", " ", x)


# Verdict strings pooled across the 26 MultiFC fact-checking sites, bucketed into three
# ordinal classes. Anything not listed (including bare numeric ratings whose
# site-specific meaning is unverified) is dropped as unmappable.
_MFC_TRUE = set(_mfc_norm_label(x) for x in [
    "true", "truth!", "verdict: true", "determination: true",
    "factscan score: true", "correct", "accurate", "verified", "fact",
    "conclusion: accurate", "mostly true", "mostly-true", "mostly truth!",
    "mostly-correct", "determination: mostly true", "promise kept",
    "in-the-green", "true messages", "authorship confirmed!",
    "confirmed authorship!", "correct attribution", "correct attribution!",
    "truth! & outdated!", "truth! & unproven!", "previously truth! now resolved!",
])
_MFC_MIXED = set(_mfc_norm_label(x) for x in [
    "half-true", "half true", "mixture", "in-between", "partly true",
    "partially true", "needs context", "not the whole story", "compromise",
    "determination: barely true", "truth! & fiction!", "truth! & misleading!",
    "truth! & disputed!", "disputed!", "unproven", "unproven!", "unverified",
    "determination: misleading", "factscan score: misleading", "misleading",
    "misleading!", "misleading recommendations", "2 pinnochios",
    "a little baloney", "some baloney", "spins the facts", "distorts the facts",
    "cherry picks", "exaggerated", "exaggerates", "understated", "half flip",
    "no flip", "conclusion: unclear", "determination: a stretch",
])
_MFC_FALSE = set(_mfc_norm_label(x) for x in [
    "false", "mostly false", "mostly-false", "pants on fire!", "pants-fire",
    "fiction!", "fiction", "mostly fiction!", "verdict: false",
    "determination: false", "factscan score: false", "conclusion: false",
    "rating: false", "we rate this claim false", "incorrect",
    "incorrect attribution!", "inaccurate attribution!", "fake", "fake news",
    "facebook scams", "scam", "scam!", "bogus warning", "virus!",
    "4 pinnochios", "3 pinnochios", "a lot of baloney", "full flop",
    "promise broken", "in-the-red", "outdated", "outdated!", "misattributed",
    "miscaptioned", "unsupported", "unsubstantiated messages",
    "verdict: unsubstantiated", "no evidence",
])
_MFC_DROP = set(_mfc_norm_label(x) for x in [
    "none", "not yet rated", "in the works", "stalled",
    "investigation pending!", "opinion!", "commentary!", "unobservable",
    "grass roots movement!", "legend", "fiction! & satire!",
    "statirical reports", "determination: huckster propaganda",
])


class MultiFCPreprocessor:
    """MultiFC front-end mirroring the Liar2Preprocessor public surface.

    The 13-column labelled MultiFC schema is:
        claim_id, claim, raw_label, url, evidence_text, category, speaker,
        checker, tags, title, published_date, claim_date, entities
    The 12-column unlabelled schema drops `raw_label`.

    Restricted-metadata text mode uses ONLY non-leaky metadata
    ({category, speaker, checker, tags, entities}); the fact-check verdict / evidence /
    title / url are never concatenated into the model text.
    """

    LABELLED_COLUMNS = [
        "claim_id", "claim", "raw_label", "url", "evidence_text", "category",
        "speaker", "checker", "tags", "title", "published_date", "claim_date",
        "entities",
    ]
    UNLABELLED_COLUMNS = [
        "claim_id", "claim", "url", "evidence_text", "category", "speaker",
        "checker", "tags", "title", "published_date", "claim_date", "entities",
    ]
    # Non-leaky metadata folded into the restricted-metadata text mode.
    SAFE_METADATA_COLUMNS = ["category", "speaker", "checker", "tags", "entities"]
    TEXT_MODES = ["claim_only", "claim_plus_safe_metadata"]

    def __init__(self, output_dir=None, min_claim_chars=1):
        self.output_dir = output_dir
        self.min_claim_chars = min_claim_chars
        self.label_order = MULTIFC_LABEL_ORDER
        self.label_to_id = MULTIFC_LABEL_TO_ID
        self._last_label_report = None

    # ---------- low-level cleaning ----------
    @staticmethod
    def _clean_text(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        x = str(x).replace("\n", " ").replace("\r", " ")
        x = re.sub(r"\s+", " ", x).strip()
        if x.lower() == "none":
            return ""
        return x

    @staticmethod
    def _clean_entities(x):
        """MultiFC `entities` is a stringified list like "['A', 'B']"; flatten to
        a readable space-joined string and drop the 'None' placeholder."""
        s = MultiFCPreprocessor._clean_text(x)
        if not s:
            return ""
        found = re.findall(r"'([^']*)'", s)
        if found:
            found = [f for f in found if f and f.lower() != "none"]
            return ", ".join(found)
        return "" if s.lower() in {"['none']", "[none]"} else s

    @classmethod
    def _map_label(cls, raw_label):
        x = _mfc_norm_label(raw_label)
        if x in _MFC_TRUE:  return "true"
        if x in _MFC_MIXED: return "mixed"
        if x in _MFC_FALSE: return "false"
        if x in _MFC_DROP:  return np.nan
        if re.fullmatch(r"\d+", x):  # bare numeric ratings: meaning unverified -> drop
            return np.nan
        return np.nan

    @staticmethod
    def _parse_date(row):
        d1 = pd.to_datetime(row.get("published_date", ""), errors="coerce", utc=True)
        if pd.notna(d1):
            return d1
        d2 = pd.to_datetime(row.get("claim_date", ""), errors="coerce", utc=True)
        if pd.notna(d2):
            return d2
        return pd.NaT

    # ---------- raw TSV loading ----------
    def _read_tsv(self, path, split_name):
        raw = pd.read_csv(path, sep="\t", header=None, dtype=str,
                          keep_default_na=False, quoting=3, engine="python")
        if raw.shape[1] == 13:
            raw.columns = self.LABELLED_COLUMNS
            raw["has_label"] = True
        elif raw.shape[1] == 12:
            raw.columns = self.UNLABELLED_COLUMNS
            raw["raw_label"] = np.nan
            raw["has_label"] = False
        else:
            raise ValueError(
                f"{path} has {raw.shape[1]} columns; expected 13 (labelled) or 12 "
                f"(unlabelled)."
            )
        raw["split"] = split_name
        raw["source_site"] = (
            raw["claim_id"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False)
            .fillna("unknown")
        )
        return raw

    def _standardize(self, df):
        out = df.copy()
        for col in ["claim", "evidence_text", "category", "speaker", "checker",
                    "tags", "title", "source_site"]:
            if col in out.columns:
                out[col] = out[col].apply(self._clean_text)
        out["entities"] = out["entities"].apply(self._clean_entities) \
            if "entities" in out.columns else ""
        out["date_parsed"] = out.apply(self._parse_date, axis=1)
        out["label_3class"] = out["raw_label"].apply(self._map_label)
        return out

    def _build_text(self, row, text_mode):
        claim = self._clean_text(row.get("claim", ""))
        if text_mode == "claim_only":
            return f"claim: {claim}"
        if text_mode == "claim_plus_safe_metadata":
            parts = [f"claim: {claim}"]
            for col in self.SAFE_METADATA_COLUMNS:
                val = self._clean_text(row.get(col, ""))
                if val:
                    parts.append(f"{col}: {val}")
            return " [SEP] ".join(parts)
        raise ValueError(f"Unknown text_mode: {text_mode}")

    # ---------- public API (mirrors Liar2Preprocessor) ----------
    def run_pipeline(self, train_path, dev_path, test_path=None):
        """Load + harmonise + clean + build text columns + chronological order.

        Only *labelled* rows are retained (official unlabelled test rows are dropped,
        since they cannot be scored). Rows with an unmappable verdict or an empty claim
        are dropped. Returns a frame carrying `label_id`, `time_rank`, and both
        `model_text__*` columns.
        """
        frames = [self._read_tsv(train_path, "train"),
                  self._read_tsv(dev_path, "dev")]
        n_official_test = 0
        if test_path:
            try:
                _te = self._read_tsv(test_path, "test")
                n_official_test = len(_te)
                # kept only for reporting; unlabelled rows are not used downstream
            except Exception as e:
                print("NOTE: could not read official test.tsv ->", repr(e))

        std = pd.concat([self._standardize(f) for f in frames], ignore_index=True)

        # label-harmonisation audit (before dropping) -> manuscript documentation
        self._last_label_report = (
            std.groupby(["raw_label", "label_3class"], dropna=False)
               .size().reset_index(name="count")
               .sort_values("count", ascending=False)
        )
        n_before = len(std)
        std = std[std["has_label"]].copy()
        std = std[std["label_3class"].isin(self.label_order)].copy()
        std["claim"] = std["claim"].apply(self._clean_text)
        std = std[std["claim"].str.len() >= self.min_claim_chars].copy()
        n_after = len(std)
        dropped_pct = 100.0 * (n_before - n_after) / max(1, n_before)

        std["label_id"] = std["label_3class"].map(self.label_to_id).astype(int)
        for mode in self.TEXT_MODES:
            std[f"model_text__{mode}"] = std.apply(
                lambda r: self._build_text(r, mode), axis=1)

        std = self.parse_and_sort_by_date(std)
        std["claim_uid"] = np.arange(len(std))
        std = self.add_time_rank(std)
        std = std.reset_index(drop=True)

        print(f"MultiFC pipeline complete.")
        print(f"  labelled rows kept : {n_after}  (dropped {n_before - n_after} "
              f"unmappable/empty of {n_before}, {dropped_pct:.1f}%)")
        if n_official_test:
            print(f"  official UNLABELLED test rows ignored (not scorable): "
                  f"{n_official_test}")
        print(f"  3-class distribution: "
              f"{std['label_3class'].value_counts().to_dict()}")
        return std

    def audit_duplicate_statements(self, df, text_col="claim"):
        """Return a table of claim strings that appear more than once."""
        key = df[text_col].astype(str).str.strip().str.lower()
        vc = key.value_counts()
        dup_keys = vc[vc > 1]
        audit = (pd.DataFrame({"claim_norm": dup_keys.index, "n_copies": dup_keys.values})
                 .reset_index(drop=True))
        print(f"Duplicate-claim audit: {len(audit)} claim strings occur >1x "
              f"({int((vc[vc > 1] - 1).sum())} redundant rows).")
        return audit

    def drop_duplicate_statements(self, df, policy="keep_first", audit_df=None,
                                  text_col="claim"):
        """De-duplicate on normalised claim text. policy='keep_first' keeps the
        earliest row (frame is date-sorted by run_pipeline before this is called)."""
        if policy != "keep_first":
            raise ValueError("only policy='keep_first' is supported")
        key = df[text_col].astype(str).str.strip().str.lower()
        keep_mask = ~key.duplicated(keep="first")
        out = df[keep_mask].copy()
        print(f"Dropped {len(df) - len(out)} duplicate-claim rows "
              f"(policy=keep_first). Remaining: {len(out)}.")
        return out

    def parse_and_sort_by_date(self, df):
        """Sort chronologically by parsed date. Rows with an unparseable date are
        placed last but keep their relative order (stable sort)."""
        out = df.copy()
        if "date_parsed" not in out.columns:
            out["date_parsed"] = out.apply(self._parse_date, axis=1)
        out["_no_date"] = out["date_parsed"].isna().astype(int)
        out = out.sort_values(["_no_date", "date_parsed"], kind="mergesort")
        out = out.drop(columns="_no_date").reset_index(drop=True)
        return out

    def add_time_rank(self, df):
        """Attach a strictly-increasing normalised time_rank in [0, 1] over the
        current row order (which run_pipeline guarantees is chronological)."""
        out = df.copy()
        n = len(out)
        out["time_rank"] = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(n)
        return out

    def label_harmonization_report(self, save_path=None):
        """Return (and optionally save) the raw->3-class mapping audit from the last
        run_pipeline call — the documentation the manuscript requires."""
        if self._last_label_report is None:
            raise RuntimeError("call run_pipeline first")
        rep = self._last_label_report.copy()
        if save_path:
            rep.to_csv(save_path, index=False)
            print("Saved label-harmonisation report:", save_path)
        return rep


print("MultiFCPreprocessor defined.")
