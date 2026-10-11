from sklearn.metrics import recall_score

def per_class_recall_rows():
    rows = []
    for scheme in SCHEMES:
        order = COLLAPSE_SCHEMES[scheme]["order"]
        store = prediction_stores[scheme]
        for (model_name, text_mode, split_mode, seed), rec in store.items():
            if text_mode not in TEXT_MODES:
                continue
            y_true = rec["y_test"]; y_pred = rec["pred"]
            r = recall_score(y_true, y_pred, average=None,
                             labels=list(range(len(order))), zero_division=0)
            for cls_id, cls_name in enumerate(order):
                rows.append({"scheme": scheme, "model": model_name,
                             "text_mode": text_mode, "seed": seed,
                             "class": cls_name, "recall": r[cls_id]})
    return pd.DataFrame(rows)

pcr = per_class_recall_rows()
pcr_summary = (pcr.groupby(["scheme", "model", "text_mode", "class"])
                  .agg(recall_mean=("recall", "mean"), recall_std=("recall", "std"))
                  .reset_index())

for _mode in TEXT_MODES:
    print("=" * 90)
    print(f"PER-CLASS RECALL (mean over seeds)   [text_mode={_mode}]")
    print("=" * 90)
    piv = (pcr_summary[pcr_summary.text_mode == _mode]
           .pivot_table(index=["scheme", "model"], columns="class",
                        values="recall_mean"))
    print(piv.to_string())
    print()

pcr_summary.to_csv(f"{OUTPUT_DIR}/collapse_per_class_recall.csv", index=False)