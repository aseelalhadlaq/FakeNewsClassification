import glob
print("Files written to", OUTPUT_DIR, ":")
for f in sorted(glob.glob(f"{OUTPUT_DIR}/**/*.csv", recursive=True)):
    print("  ", f)

print("""
Artifact guide
--------------
collapse_main_results.csv        Every (scheme x model x seed) run, full metric row.
collapse_summary_by_model.csv    Macro-F1 mean +/- std per scheme/model (Section 6).
collapse_effect_size_*.csv       DA-MiniLM-RWAC minus MiniLM-LogReg-RW, abs diff +
                                 95% bootstrap CI per granularity (the headline table).
collapse_bootstrap_results.csv   Per-seed paired-bootstrap detail.
collapse_per_class_recall.csv    Per-class recall per scheme/model (minority-collapse
                                 diagnostic).
""")