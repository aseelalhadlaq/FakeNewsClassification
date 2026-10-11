# ============================================================
# 15. Final report export and zip
# ============================================================

summary_lines = []
summary_lines.append("MultiFC DA-MiniLM++ Temporal Robustness Experiment")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append("Main model:")
summary_lines.append("DA-MiniLM++ = frozen MiniLM embeddings + class-balanced weighting + recency weighting + validation calibration.")
summary_lines.append("")
summary_lines.append("Dataset note:")
summary_lines.append("MultiFC labels are harmonized into a transparent 3-class veracity space: false, mixed, true.")
summary_lines.append("Evidence text and fact-check article titles are excluded by default to reduce leakage risk.")
summary_lines.append("")
summary_lines.append("Evaluation protocols:")
summary_lines.append("1. official_train_to_dev: inner validation from train only; dev is held out for evaluation.")
summary_lines.append("2. internal_chronological: 70/15/15 chronological split on train+dev labelled rows.")
summary_lines.append("3. internal_random_stratified: 70/15/15 random stratified split on train+dev labelled rows.")
summary_lines.append("4. rolling_window: expanding temporal windows for DA-MiniLM++.")
summary_lines.append("")
summary_lines.append("Files exported:")
for p in sorted(OUTPUT_DIR.glob("*.csv")):
    summary_lines.append(f"- {p.name}")

if "summary_df" in globals() and len(summary_df) > 0:
    summary_lines.append("")
    summary_lines.append("Main mean/std summary:")
    summary_lines.append("-" * 80)
    summary_lines.append(summary_df.to_string(index=False))

if "bootstrap_df" in globals() and len(bootstrap_df) > 0:
    summary_lines.append("")
    summary_lines.append("Bootstrap tests with BH correction:")
    summary_lines.append("-" * 80)
    summary_lines.append(bootstrap_df.to_string(index=False))

if "rolling_summary_df" in globals() and len(rolling_summary_df) > 0:
    summary_lines.append("")
    summary_lines.append("Rolling-window summary:")
    summary_lines.append("-" * 80)
    summary_lines.append(rolling_summary_df.to_string(index=False))

summary_text = "\n".join(summary_lines)
summary_path = OUTPUT_DIR / "experiment_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(summary_text)

zip_path = shutil.make_archive(str(OUTPUT_DIR), "zip", OUTPUT_DIR)
print("Created zip:", zip_path)