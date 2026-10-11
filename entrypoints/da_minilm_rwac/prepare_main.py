# ============================================================
# SECTION 6: DA-MiniLM-RWAC / MiniLM-LogReg-RW -- main model
# battery on the FINAL DEDUPLICATED LIAR2 cohort (N = 22,923).
# ============================================================

#from temporal_splitter import TemporalSplitter
#from da_minilm_model import DAMiniLMModel
#from da_minilm_experiment_runner import DAMiniLMExperimentRunner

liar2_final_df = liar2_df.copy().reset_index(drop=True)
assert len(liar2_final_df) == 22923

da_model = DAMiniLMModel(output_dir=str(OUTPUT_ROOT / "da_minilm"), num_classes=6)
da_runner = DAMiniLMExperimentRunner(splitter=liar2_prep, model=da_model, num_classes=6)

# Embeddings, once per text mode (cached to disk automatically by DAMiniLMModel).
da_embeddings = {
    mode: da_model.encode_texts_minilm(liar2_final_df[f"model_text__{mode}"].tolist(), cache_key=f"{mode}_n{len(liar2_final_df)}")
    for mode in TEXT_MODES
}

