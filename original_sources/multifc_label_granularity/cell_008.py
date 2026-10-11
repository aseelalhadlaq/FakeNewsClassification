# ============================================================
# 1b. Label-collapse definitions
# ============================================================
# 3-class identity (native MultiFC granularity)
THREECLASS_IDENTITY = {0: 0, 1: 1, 2: 2}
THREECLASS_ORDER    = ["false", "mixed", "true"]

# 2-class: {false, true}. mixed -> false (conservative veracity screen).
THREECLASS_TO_TWO   = {0: 0, 1: 0, 2: 1}
TWOCLASS_ORDER      = ["false", "true"]

COLLAPSE_SCHEMES = {
    "3class": {"map": THREECLASS_IDENTITY, "order": THREECLASS_ORDER, "num_classes": 3},
    "2class": {"map": THREECLASS_TO_TWO,   "order": TWOCLASS_ORDER,   "num_classes": 2},
}


class LabelCollapser:
    """Remaps an already-cleaned MultiFC DataFrame's `label_id` to a coarser
    granularity, leaving claim text, date order, time_rank and the restricted-metadata
    text modes untouched. Native 3-class labels are retained as `label_id_3class`."""

    def __init__(self, scheme: str):
        if scheme not in COLLAPSE_SCHEMES:
            raise ValueError(f"scheme must be one of {list(COLLAPSE_SCHEMES)}, "
                             f"got {scheme!r}")
        cfg = COLLAPSE_SCHEMES[scheme]
        self.scheme = scheme
        self.map = cfg["map"]
        self.label_order = cfg["order"]
        self.num_classes = cfg["num_classes"]

    def transform(self, df: pd.DataFrame, label_id_col: str = "label_id") -> pd.DataFrame:
        if label_id_col not in df.columns:
            raise KeyError(f"{label_id_col!r} not in DataFrame columns")
        out = df.copy()
        three = out[label_id_col].astype(int)
        unknown = set(three.unique()) - set(self.map)
        if unknown:
            raise ValueError(f"label_id values {sorted(unknown)} absent from "
                             f"{self.scheme} map (expected 0..2)")
        out["label_id_3class"] = three.values
        out[label_id_col] = three.map(self.map).astype(int).values
        out["label_collapsed"] = out[label_id_col].map(
            {i: n for i, n in enumerate(self.label_order)}).values
        return out

    def describe_collapse(self, df: pd.DataFrame,
                          label_id_col: str = "label_id") -> pd.DataFrame:
        three_col = "label_id_3class" if "label_id_3class" in df.columns else label_id_col
        rows = []
        for sid in range(3):
            n = int((df[three_col] == sid).sum())
            cid = self.map[sid]
            rows.append({"three_id": sid, "three_label": THREECLASS_ORDER[sid],
                         "collapsed_id": cid,
                         "collapsed_label": self.label_order[cid], "n_rows": n})
        tbl = pd.DataFrame(rows)
        totals = (tbl.groupby(["collapsed_id", "collapsed_label"])["n_rows"].sum()
                     .reset_index().rename(columns={"n_rows": "collapsed_total"}))
        return tbl.merge(totals, on=["collapsed_id", "collapsed_label"])


# quick self-test on synthetic ids
_demo = pd.DataFrame({"label_id": [0, 1, 2]})
for _s in COLLAPSE_SCHEMES:
    _c = LabelCollapser(_s).transform(_demo)
    print(f"{_s:8s} -> {_c['label_id'].tolist()}  (K={LabelCollapser(_s).num_classes})")
