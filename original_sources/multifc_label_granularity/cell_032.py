print("Files written to", OUTPUT_DIR, ":")
for f in sorted(glob.glob(f"{OUTPUT_DIR}/**/*.csv", recursive=True)):
    print("  ", f)

print("""
Artifact guide
--------------
multifc_label_harmonization_report.csv  Raw MultiFC verdict -> 3-class mapping audit
                                        (26-site taxonomy; the manuscript documentation).
collapse_main_results.csv               Every (scheme x model x seed) run, full metrics.
collapse_summary_by_model.csv           Macro-F1 mean +/- std per scheme/model (Sec 5).
collapse_effect_size_*.csv              DA-MiniLM-RWAC minus MiniLM-LogReg-RW, abs diff +
                                        95% bootstrap CI per granularity (headline).
collapse_bootstrap_results.csv          Per-seed paired-bootstrap detail.
collapse_seed_level_effect.csv          Between-seed effect size + t-interval (Sec 5b).
collapse_per_class_recall.csv           Per-class recall per scheme/model (minority
                                        collapse diagnostic).
collapse_chance_floors.csv              Stratified/uniform/majority macro-F1 floors.
collapse_margin_over_chance.csv         Each model's margin over the stratified floor.
""")
