from scipy import stats

def seed_level_effect(main_results, model_a="DA-MiniLM-RWAC",
                      model_b="MiniLM-LogReg-RW",
                      text_mode=TEXT_MODE, split_mode=SPLIT_MODE):
    rows = []
    for scheme in SCHEMES:
        sub = main_results[(main_results.scheme == scheme) &
                           (main_results.text_mode == text_mode) &
                           (main_results.split_mode == split_mode)]
        a = sub[sub.model == model_a].sort_values("seed")["macro_f1"].values
        b = sub[sub.model == model_b].sort_values("seed")["macro_f1"].values
        if len(a) == 0 or len(b) == 0:
            continue
        n = min(len(a), len(b))
        diffs = a[:n] - b[:n]
        mean = diffs.mean(); sd = diffs.std(ddof=1) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 1 else 0.0
        tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
        ci_l, ci_u = mean - tcrit * se, mean + tcrit * se
        t_stat, p_t = stats.ttest_1samp(diffs, 0.0) if n > 1 else (np.nan, np.nan)
        n_pos = int((diffs > 0).sum())
        p_sign = stats.binomtest(n_pos, n, 0.5).pvalue
        rows.append({"scheme": scheme, "K": COLLAPSE_SCHEMES[scheme]["num_classes"],
                     "n_seeds": n, "mean_diff": round(mean, 4),
                     "between_seed_sd": round(sd, 4),
                     "t_CI95_low": round(ci_l, 4), "t_CI95_high": round(ci_u, 4),
                     "t_p": round(p_t, 4) if p_t == p_t else np.nan,
                     "seeds_favoring_A": f"{n_pos}/{n}",
                     "sign_p": round(p_sign, 4)})
    df = pd.DataFrame(rows)
    _r = {"3class": 0, "2class": 1}
    return df.assign(_r=df["scheme"].map(_r)).sort_values("_r").drop(columns="_r")

seed_effect = seed_level_effect(main_results)
print("=" * 96)
print(f"SEED-LEVEL EFFECT SIZE  DA-MiniLM-RWAC minus MiniLM-LogReg-RW  "
      f"[text_mode={TEXT_MODE}]")
print("=" * 96)
print(seed_effect.to_string(index=False))
seed_effect.to_csv(f"{OUTPUT_DIR}/collapse_seed_level_effect.csv", index=False)

print("""
Reading guide
-------------
- mean_diff + t_CI95: primary. If the interval excludes 0, the advantage is
  detectable at that granularity across seeds.
- seeds_favoring_A: corroboration. At n=5 the sign test's p-value floor is 0.0625,
  so it never reaches 0.05 here by design.
- Agreement between this between-seed interval and the Section 5 within-test bootstrap
  is the robust external-replication claim.
""")
