# ---- Headline effect-size table: DA-MiniLM-RWAC minus MiniLM-LogReg-RW ----
def effect_table(text_mode):
    rows = []
    for scheme in SCHEMES:
        K = COLLAPSE_SCHEMES[scheme]["num_classes"]
        sub = summary[(summary.scheme == scheme) & (summary.text_mode == text_mode)]
        try:
            f_rwac = sub.loc[sub.model == "DA-MiniLM-RWAC", "macro_f1_mean"].iloc[0]
            f_rw   = sub.loc[sub.model == "MiniLM-LogReg-RW", "macro_f1_mean"].iloc[0]
        except IndexError:
            continue
        abs_diff = f_rwac - f_rw
        rel = 100.0 * abs_diff / f_rw if f_rw else np.nan
        ci_l = ci_u = p = np.nan
        sig = None
        if len(bootstrap_results):
            b = bootstrap_results[(bootstrap_results.scheme == scheme) &
                                  (bootstrap_results.text_mode == text_mode)]
            if len(b):
                ci_l = b["ci_lower_95"].mean(); ci_u = b["ci_upper_95"].mean()
                p = b["p_value"].mean()
                sig = bool((b["significant_at_95"]).mean() >= 0.5)
        rows.append({"scheme": scheme, "K": K,
                     "RWAC_macroF1": round(f_rwac, 4),
                     "LogRegRW_macroF1": round(f_rw, 4),
                     "abs_diff": round(abs_diff, 4),
                     "rel_pct": round(rel, 2) if rel == rel else np.nan,
                     "boot_CI95_low": round(ci_l, 4) if ci_l == ci_l else np.nan,
                     "boot_CI95_high": round(ci_u, 4) if ci_u == ci_u else np.nan,
                     "boot_p": round(p, 4) if p == p else np.nan,
                     "sig_at_95": sig})
    return pd.DataFrame(rows)

for _mode in TEXT_MODES:
    print("=" * 90)
    print(f"HEADLINE EFFECT SIZE  DA-MiniLM-RWAC  minus  MiniLM-LogReg-RW   "
          f"[text_mode={_mode}]")
    print("(absolute macro-F1 difference and 95% bootstrap CI reported before "
          "significance)")
    print("=" * 90)
    et = effect_table(_mode)
    print(et.to_string(index=False))
    et.to_csv(f"{OUTPUT_DIR}/collapse_effect_size_{_mode}.csv", index=False)
    print()
