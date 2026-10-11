# ============================================================
# 13. Rolling-window temporal evaluation for DA-MiniLM++
# ============================================================

rolling_rows = []

if RUN_ROLLING_WINDOWS and RUN_DA_MINILM_PLUS and len(rolling_windows) > 0:
    ROLLING_TEXT_MODE = "claim_plus_safe_metadata"
    ROLLING_SEEDS = SEEDS
    X = all_embeddings[ROLLING_TEXT_MODE]

    for w in rolling_windows:
        for seed in ROLLING_SEEDS:
            print(f"\n[Rolling DA-MiniLM++] window={w['window_id']} seed={seed}")
            row, lambda_df, pred, proba = run_single_minilm_experiment(
                X, y_all, time_rank_all,
                w["train_idx"], w["valid_idx"], w["test_idx"],
                text_mode=ROLLING_TEXT_MODE,
                eval_setup="rolling_window",
                seed=seed,
                model_kind="da_minilm_mlp",
            )
            row.update({
                "window_id": w["window_id"],
                "train_end_frac": w["train_end_frac"],
                "valid_end_frac": w["valid_end_frac"],
                "test_end_frac": w["test_end_frac"],
            })
            rolling_rows.append(row)
            print(f"  macro_f1={row['macro_f1']:.4f}, acc={row['accuracy']:.4f}, lambda={row['recency_lambda']}")

rolling_results_df = pd.DataFrame(rolling_rows)
rolling_path = OUTPUT_DIR / "rolling_window_da_minilm_plus_results.csv"
rolling_results_df.to_csv(rolling_path, index=False)
print("Saved:", rolling_path)

if len(rolling_results_df) > 0:
    rolling_summary_df = summarize_results(
        rolling_results_df,
        group_cols=["window_id", "model", "text_mode", "eval_setup"]
    )
    rolling_summary_path = OUTPUT_DIR / "rolling_window_da_minilm_plus_summary.csv"
    rolling_summary_df.to_csv(rolling_summary_path, index=False)
    print("Saved:", rolling_summary_path)
    display(rolling_summary_df)

    plt.figure(figsize=(8, 4))
    plot_df = rolling_summary_df.sort_values("window_id")
    plt.plot(plot_df["window_id"], plot_df["macro_f1_mean"], marker="o")
    plt.xlabel("Rolling window ID")
    plt.ylabel("Mean macro-F1")
    plt.title("DA-MiniLM++ rolling-window temporal robustness on MultiFC")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "rolling_window_macro_f1.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print("Saved:", fig_path)
else:
    print("No rolling-window results generated.")