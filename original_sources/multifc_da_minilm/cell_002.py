# ============================================================
# 1. Imports and global configuration
# ============================================================
import os, re, json, math, random, warnings, shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, log_loss
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# -----------------------------
# Experiment switches
# -----------------------------
RUN_DA_MINILM_PLUS = True
RUN_LOGREG_BASELINE = True
RUN_ROLLING_WINDOWS = True
RUN_OFFICIAL_TEST_PREDICTIONS = True

# Keep title/evidence disabled by default to reduce leakage.
TEXT_MODES = ["claim_only", "claim_plus_safe_metadata"]
EVAL_SETUPS = ["official_train_to_dev", "internal_chronological", "internal_random_stratified"]

SEEDS = [42, 43, 44, 45, 46]
RECENCY_LAMBDAS = [0.00, 0.05, 0.10, 0.20, 0.30]
TAU_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]
TEMPERATURE_GRID = [0.75, 1.0, 1.25, 1.5, 2.0]

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

OUTPUT_DIR = Path("/kaggle/working/multifc_da_minilm_plus_v1_0")
if not Path("/kaggle").exists():
    OUTPUT_DIR = Path("/mnt/data/multifc_da_minilm_plus_v1_0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Output directory:", OUTPUT_DIR)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True