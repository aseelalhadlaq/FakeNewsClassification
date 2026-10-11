# ============================================================
# 0b. Imports and experiment configuration
# ============================================================
import os, re, json, math, random, warnings, glob
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, log_loss,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ----- EDIT THESE THREE PATHS to match your Kaggle MultiFC dataset -----
# The loader also auto-searches common Kaggle input locations if these are wrong.
MULTIFC_TRAIN = "/kaggle/input/datasets/badalahmmed/multifc/train.tsv"
MULTIFC_DEV   = "/kaggle/input/datasets/badalahmmed/multifc/dev.tsv"
MULTIFC_TEST  = "/kaggle/input/datasets/badalahmmed/multifc/test.tsv"   # unlabelled; used only for row-count reporting

OUTPUT_DIR = "/kaggle/working/multifc_label_collapse_results"
if not Path("/kaggle").exists():
    OUTPUT_DIR = "/mnt/data/multifc_label_collapse_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Experiment configuration (mirrors the LIAR2 collapse notebook) -----
# Restricted-metadata text mode is the paper's headline condition.
# (In-code the column is named "claim_plus_safe_metadata"; the paper calls it
#  "restricted metadata" — same thing.)
TEXT_MODE   = "claim_plus_safe_metadata"
SPLIT_MODE  = "chronological"          # headline split
SEEDS       = [0, 1, 2, 3, 4]          # 5-seed battery
N_BOOT      = 2000                     # paired-bootstrap resamples

# MultiFC has no 6-class ordinal labels. Native granularity is 3-class; 2-class is
# derived from it (mixed -> false). The 6-class scheme is intentionally absent.
SCHEMES = ["3class", "2class"]

# Set True to also run the plain claim-only text mode at each granularity.
ALSO_RUN_CLAIM_ONLY = False

# MiniLM / MLP hyperparameters (identical to the LIAR2 main model)
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE  = 128
MLP_EPOCHS        = 35
MLP_BATCH_SIZE    = 128
MLP_LR            = 1e-3
MLP_WEIGHT_DECAY  = 1e-4
MLP_PATIENCE      = 6
RECENCY_LAMBDAS   = [0.0, 0.05, 0.10, 0.20, 0.30]
CLASS_WEIGHT_BETA = 0.999
TEMPERATURE_GRID  = [0.75, 1.0, 1.25, 1.5, 2.0]
TAU_GRID          = [0.0, 0.5, 1.0, 1.5, 2.0]

print("Config loaded. Output dir:", OUTPUT_DIR)
print("Schemes:", SCHEMES, "| headline text mode:", TEXT_MODE, "| split:", SPLIT_MODE)
