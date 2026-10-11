#from drift_analysis import DriftAnalysis

da = DriftAnalysis()

summary_df = da.run_all_drift_experiments(
    train_df,
    test_df,
    include_embedding_shift=True,  # set True only if this notebook has internet enabled
)

summary_df