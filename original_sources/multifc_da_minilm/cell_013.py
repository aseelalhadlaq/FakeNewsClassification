# ============================================================
# 12. Classification reports and confusion matrices for best conditions
# ============================================================

# Save one classification report per model/text/eval/seed for transparent inspection.
report_rows = []
for key, payload in prediction_store.items():
    model, text_mode, eval_setup, seed = key
    y_test = payload["y_test"]
    pred = payload["pred"]
    report = classification_report(
        y_test,
        pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    for label in LABEL_ORDER:
        report_rows.append({
            "model": model,
            "text_mode": text_mode,
            "eval_setup": eval_setup,
            "seed": seed,
            "label": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1_score": report[label]["f1-score"],
            "support": report[label]["support"],
        })

class_report_df = pd.DataFrame(report_rows)
class_report_path = OUTPUT_DIR / "per_class_reports.csv"
class_report_df.to_csv(class_report_path, index=False)
print("Saved:", class_report_path)
display(class_report_df.head())

# Confusion matrices for seed 42 under official evaluation.
for model in ["DA-MiniLM++", "MiniLM-LogReg-RW"]:
    for text_mode in TEXT_MODES:
        key = (model, text_mode, "official_train_to_dev", 42)
        if key not in prediction_store:
            continue
        y_test = prediction_store[key]["y_test"]
        pred = prediction_store[key]["pred"]
        cm = confusion_matrix(y_test, pred, labels=list(range(NUM_CLASSES)))
        cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in LABEL_ORDER], columns=[f"pred_{x}" for x in LABEL_ORDER])
        cm_path = OUTPUT_DIR / f"confusion_matrix_{safe_cache_name(model)}_{text_mode}_official_seed42.csv"
        cm_df.to_csv(cm_path)
        print("Saved:", cm_path)
        display(cm_df)