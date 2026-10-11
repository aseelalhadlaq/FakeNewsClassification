# ============================================================
# 10. Run main DA-MiniLM++ and MiniLM-LogReg experiments
# ============================================================

main_rows = []

for text_mode in TEXT_MODES:
    X = all_embeddings[text_mode]
    for eval_setup in EVAL_SETUPS:
        for seed in SEEDS:
            train_idx, valid_idx, test_idx = get_eval_indices(eval_setup, seed=seed)
            print(f"\n=== text_mode={text_mode} | eval_setup={eval_setup} | seed={seed} ===")
            print(f"n_train={len(train_idx)}, n_valid={len(valid_idx)}, n_test={len(test_idx)}")

            if RUN_LOGREG_BASELINE:
                row, lambda_df, pred, proba = run_single_minilm_experiment(
                    X, y_all, time_rank_all, train_idx, valid_idx, test_idx,
                    text_mode=text_mode,
                    eval_setup=eval_setup,
                    seed=seed,
                    model_kind="minilm_logreg",
                )
                main_rows.append(row)
                lambda_selection_rows.extend(lambda_df.to_dict("records"))
                print(f"  MiniLM-LogReg-RW macro_f1={row['macro_f1']:.4f}, acc={row['accuracy']:.4f}")

            if RUN_DA_MINILM_PLUS:
                row, lambda_df, pred, proba = run_single_minilm_experiment(
                    X, y_all, time_rank_all, train_idx, valid_idx, test_idx,
                    text_mode=text_mode,
                    eval_setup=eval_setup,
                    seed=seed,
                    model_kind="da_minilm_mlp",
                )
                main_rows.append(row)
                lambda_selection_rows.extend(lambda_df.to_dict("records"))
                print(f"  DA-MiniLM++ macro_f1={row['macro_f1']:.4f}, acc={row['accuracy']:.4f}, ece={row['ece']:.4f}")

main_results_df = pd.DataFrame(main_rows)
main_results_path = OUTPUT_DIR / "main_results_per_seed.csv"
main_results_df.to_csv(main_results_path, index=False)
print("Saved:", main_results_path)
display(main_results_df.head())

lambda_selection_df = pd.DataFrame(lambda_selection_rows)
lambda_path = OUTPUT_DIR / "lambda_selection_records.csv"
lambda_selection_df.to_csv(lambda_path, index=False)
print("Saved:", lambda_path)

summary_df = summarize_results(main_results_df, group_cols=["model", "text_mode", "eval_setup"])
summary_df = summary_df.sort_values(["eval_setup", "text_mode", "macro_f1_mean"], ascending=[True, True, False])
summary_path = OUTPUT_DIR / "main_results_mean_std_summary.csv"
summary_df.to_csv(summary_path, index=False)
print("Saved:", summary_path)
display(summary_df)