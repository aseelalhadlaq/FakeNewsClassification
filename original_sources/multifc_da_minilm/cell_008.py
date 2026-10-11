# ============================================================
# 7. Frozen MiniLM embedding extraction
# ============================================================

embedding_cache = {}
encoder = SentenceTransformer(MINILM_MODEL_NAME, device=DEVICE)


def safe_cache_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def encode_texts_minilm(texts, cache_key, batch_size=EMBED_BATCH_SIZE):
    cache_key = safe_cache_name(cache_key)
    cache_path = OUTPUT_DIR / f"embeddings_{cache_key}.npy"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    if cache_path.exists():
        arr = np.load(cache_path)
        embedding_cache[cache_key] = arr
        print("Loaded cached embeddings:", cache_path, arr.shape)
        return arr
    arr = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(cache_path, arr)
    embedding_cache[cache_key] = arr
    print("Saved embeddings:", cache_path, arr.shape)
    return arr

all_embeddings = {}
for mode in TEXT_MODES:
    text_col = f"text__{mode}"
    all_embeddings[mode] = encode_texts_minilm(labelled_all[text_col].fillna("").tolist(), f"labelled_all_{mode}")

# Encode official unlabelled test text only when needed.
test_embeddings = {}
if RUN_OFFICIAL_TEST_PREDICTIONS and len(test_unlabelled) > 0:
    for mode in TEXT_MODES:
        text_col = f"text__{mode}"
        test_embeddings[mode] = encode_texts_minilm(test_unlabelled[text_col].fillna("").tolist(), f"official_test_unlabelled_{mode}")

y_all = labelled_all["label_id"].astype(int).values
time_rank_all = labelled_all["time_rank"].astype(float).values
print("Embedding modes:", {k: v.shape for k, v in all_embeddings.items()})