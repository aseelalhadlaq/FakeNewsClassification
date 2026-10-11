pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

summary = (main_results
    .groupby(["scheme", "num_classes", "model", "text_mode"])
    .agg(macro_f1_mean=("macro_f1", "mean"),
         macro_f1_std =("macro_f1", "std"),
         accuracy_mean=("accuracy", "mean"),
         macro_recall_mean=("macro_recall", "mean"),
         ece_mean=("ece", "mean"),
         n_runs=("macro_f1", "count"))
    .reset_index())

_scheme_rank = {"3class": 0, "2class": 1}
summary["_r"] = summary["scheme"].map(_scheme_rank)
summary = summary.sort_values(["_r", "text_mode", "macro_f1_mean"],
                              ascending=[True, True, False]).drop(columns="_r")

print("=" * 90)
print("PER-MODEL SUMMARY  (macro-F1 mean +/- std across seeds)")
print("=" * 90)
print(summary.to_string(index=False))
summary.to_csv(f"{OUTPUT_DIR}/collapse_summary_by_model.csv", index=False)
