# 3-class: {false, mixed, true} -- identical taxonomy to collapsed MultiFC
SIXCLASS_TO_THREE = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
THREECLASS_ORDER  = ["false", "mixed", "true"]

# 2-class: {false, true}, half-true on the TRUE side (LIAR convention)
SIXCLASS_TO_TWO   = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
TWOCLASS_ORDER    = ["false", "true"]

# Robustness variant: half-true on the FALSE side
SIXCLASS_TO_TWO_HT_FALSE = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1}

# Identity (reference 6-class run)
SIXCLASS_IDENTITY = {i: i for i in range(6)}
SIXCLASS_ORDER    = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

COLLAPSE_SCHEMES = {
    "6class":                {"map": SIXCLASS_IDENTITY,        "order": SIXCLASS_ORDER,   "num_classes": 6},
    "3class":                {"map": SIXCLASS_TO_THREE,        "order": THREECLASS_ORDER, "num_classes": 3},
    "2class":                {"map": SIXCLASS_TO_TWO,          "order": TWOCLASS_ORDER,   "num_classes": 2},
    "2class_halftrue_false": {"map": SIXCLASS_TO_TWO_HT_FALSE, "order": TWOCLASS_ORDER,   "num_classes": 2},
}


class LabelCollapser:
    """Remaps an already-cleaned LIAR2 DataFrame's `label_id` to a coarser
    granularity, leaving statement text, date order, time_rank and the
    restricted-metadata text modes untouched. The original 6-class labels are
    retained as `label_id_6class` for auditing."""

    def __init__(self, scheme: str):
        if scheme not in COLLAPSE_SCHEMES:
            raise ValueError(f"scheme must be one of {list(COLLAPSE_SCHEMES)}, got {scheme!r}")
        cfg = COLLAPSE_SCHEMES[scheme]
        self.scheme = scheme
        self.map = cfg["map"]
        self.label_order = cfg["order"]
        self.num_classes = cfg["num_classes"]

    def transform(self, df: pd.DataFrame, label_id_col: str = "label_id") -> pd.DataFrame:
        if label_id_col not in df.columns:
            raise KeyError(f"{label_id_col!r} not in DataFrame columns")
        out = df.copy()
        six = out[label_id_col].astype(int)
        unknown = set(six.unique()) - set(self.map)
        if unknown:
            raise ValueError(f"label_id values {sorted(unknown)} absent from {self.scheme} map (expected 0..5)")
        out["label_id_6class"] = six.values
        out[label_id_col] = six.map(self.map).astype(int).values
        out["label_collapsed"] = out[label_id_col].map(
            {i: n for i, n in enumerate(self.label_order)}).values
        return out

    def describe_collapse(self, df: pd.DataFrame, label_id_col: str = "label_id") -> pd.DataFrame:
        six_col = "label_id_6class" if "label_id_6class" in df.columns else label_id_col
        rows = []
        for sid in range(6):
            n = int((df[six_col] == sid).sum())
            cid = self.map[sid]
            rows.append({"six_id": sid, "six_label": SIXCLASS_ORDER[sid],
                         "collapsed_id": cid, "collapsed_label": self.label_order[cid],
                         "n_rows": n})
        tbl = pd.DataFrame(rows)
        totals = (tbl.groupby(["collapsed_id", "collapsed_label"])["n_rows"].sum()
                     .reset_index().rename(columns={"n_rows": "collapsed_total"}))
        return tbl.merge(totals, on=["collapsed_id", "collapsed_label"])


# quick self-test on synthetic ids
_demo = pd.DataFrame({"label_id": list(range(6))})
for _s in COLLAPSE_SCHEMES:
    _c = LabelCollapser(_s).transform(_demo)
    print(f"{_s:24s} -> {_c['label_id'].tolist()}  (K={LabelCollapser(_s).num_classes})")