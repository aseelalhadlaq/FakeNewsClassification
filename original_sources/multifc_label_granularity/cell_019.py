# A shared model instance ONLY for encoding (num_classes irrelevant for encode).
_encoder = DAMiniLMModel(output_dir=OUTPUT_DIR, num_classes=3)

_text_modes_to_embed = [TEXT_MODE] + (["claim_only"] if ALSO_RUN_CLAIM_ONLY else [])

embeddings = {}
for _mode in _text_modes_to_embed:
    _col = f"model_text__{_mode}"
    print(f"Encoding text mode: {_mode}  ({_col})")
    embeddings[_mode] = _encoder.encode_texts_minilm(
        df3[_col].tolist(), cache_key=f"multifc_collapse_{_mode}")
    print(f"  -> embedding shape: {embeddings[_mode].shape}")

TEXT_MODES = list(embeddings.keys())
print("\nEmbeddings ready for text modes:", TEXT_MODES)
