from sklearn.metrics import f1_score as _f1

def compute_chance_floors(labels_by_scheme, n_sim=2000, seed=0):
    rows = []
    for scheme, y in labels_by_scheme.items():
        y = np.asarray(y).astype(int)
        K = int(y.max()) + 1
        counts = np.bincount(y, minlength=K).astype(float)
        p = counts / counts.sum()
        rng = np.random.RandomState(seed)
        maj = np.full(len(y), int(p.argmax()))
        maj_acc = float((maj == y).mean())
        maj_f1 = _f1(y, maj, average="macro", zero_division=0)
        strat_f1 = np.empty(n_sim); unif_f1 = np.empty(n_sim)
        for b in range(n_sim):
            strat_f1[b] = _f1(y, rng.choice(K, size=len(y), p=p),
                              average="macro", zero_division=0)
            unif_f1[b]  = _f1(y, rng.randint(0, K, size=len(y)),
                              average="macro", zero_division=0)
        rows.append({"scheme": scheme, "K": K, "N_test": int(counts.sum()),
                     "majority_prevalence": round(p.max(), 4),
                     "uniform_acc_1overK": round(1.0 / K, 4),
                     "uniform_macroF1": round(unif_f1.mean(), 4),
                     "stratified_acc": round(float((p ** 2).sum()), 4),
                     "stratified_macroF1": round(strat_f1.mean(), 4),
                     "stratified_macroF1_p95": round(np.percentile(strat_f1, 95), 4),
                     "majority_acc": round(maj_acc, 4),
                     "majority_macroF1": round(maj_f1, 4)})
    return pd.DataFrame(rows)

_splitter_ref = TemporalSplitter(train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15)
_test_labels_by_scheme = {}
for scheme in SCHEMES:
    df_c = LabelCollapser(scheme).transform(df3).reset_index(drop=True)
    _tr, _va, _te = _splitter_ref.get_split(df_c, SPLIT_MODE, seed=SEEDS[0])
    _test_labels_by_scheme[scheme] = df_c["label_id"].values[_te]

floors = compute_chance_floors(_test_labels_by_scheme, n_sim=2000, seed=0)

_scheme_rank = {"3class": 0, "2class": 1}
obs = (main_results.groupby(["scheme", "model"])["macro_f1"].mean().reset_index())
margin_rows = []
for _, fr in floors.iterrows():
    s = fr["scheme"]; floor = fr["stratified_macroF1"]
    for model in ["DA-MiniLM-RWAC", "MiniLM-LogReg-RW", "Uniform-LR"]:
        m = obs[(obs.scheme == s) & (obs.model == model)]
        if len(m):
            mf = float(m["macro_f1"].iloc[0])
            margin_rows.append({"scheme": s, "K": int(fr["K"]), "model": model,
                                "macro_f1": round(mf, 4),
                                "stratified_floor": floor,
                                "margin_over_chance": round(mf - floor, 4)})
margin_df = pd.DataFrame(margin_rows)
margin_df["_r"] = margin_df["scheme"].map(_scheme_rank)
margin_df = margin_df.sort_values(["_r", "macro_f1"],
                                  ascending=[True, False]).drop(columns="_r")

pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
print("=" * 96)
print("CHANCE-LEVEL FLOORS PER GRANULARITY (macro-F1 chance is NOT 1/K)")
print("=" * 96)
floors["_r"] = floors["scheme"].map(_scheme_rank)
print(floors.sort_values("_r").drop(columns="_r").to_string(index=False))

print("\n" + "=" * 96)
print("MARGIN OVER CHANCE  (raw macro-F1 is NOT comparable across K)")
print("=" * 96)
print(margin_df.to_string(index=False))

floors.drop(columns="_r").to_csv(f"{OUTPUT_DIR}/collapse_chance_floors.csv", index=False)
margin_df.to_csv(f"{OUTPUT_DIR}/collapse_margin_over_chance.csv", index=False)
