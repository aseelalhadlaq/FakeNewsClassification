# ============================================================
# SECTION 6: DA-MiniLM-RWAC / MiniLM-LogReg-RW -- the main model
# battery, on the FULL (non-deduplicated) LIAR2 pipeline output,
# matching da-minilm-main-model-v1-0.ipynb exactly (that notebook does
# not deduplicate before its own chronological/random splits).
# ============================================================

#from temporal_splitter import TemporalSplitter
#from da_minilm_model import DAMiniLMModel
#from da_minilm_experiment_runner import DAMiniLMExperimentRunner

liar2_full_df = liar2_df.reset_index(drop=True)
liar2_full_df = liar2_prep.add_time_rank(liar2_full_df)

da_model = DAMiniLMModel(output_dir=str(OUTPUT_ROOT / "da_minilm"), num_classes=6)
da_runner = DAMiniLMExperimentRunner(splitter=liar2_prep, model=da_model, num_classes=6)

# Embeddings, once per text mode (cached to disk automatically by DAMiniLMModel).
da_embeddings = {
    mode: da_model.encode_texts_minilm(liar2_full_df[f"model_text__{mode}"].tolist(), cache_key=mode)
    for mode in TEXT_MODES
}

