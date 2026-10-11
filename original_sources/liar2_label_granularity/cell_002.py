import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

# ----- EDIT THESE THREE PATHS to match your Kaggle LIAR2 dataset -----
LIAR2_TRAIN = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/train.csv"
LIAR2_VALID = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/valid.csv"
LIAR2_TEST  = "/kaggle/input/datasets/badalahmmed/liar-dataset/liar2/liar2/test.csv"

OUTPUT_DIR = "/kaggle/working/label_collapse_results"

# ----- Experiment configuration -----
# Restricted-metadata text mode is the paper's headline condition.
# (In-code the column is named "claim_plus_safe_metadata"; the paper calls it
#  "restricted metadata" — same thing.)
TEXT_MODE   = "claim_plus_safe_metadata"
SPLIT_MODE  = "chronological"          # the paper's headline split
SEEDS       = [0, 1, 2, 3, 4]          # 5-seed battery
N_BOOT      = 2000                     # paired-bootstrap resamples

# Which collapse schemes to run. "6class" is included as the reference point.
SCHEMES = ["6class", "3class", "2class", "2class_halftrue_false"]

# Set True to also run the plain claim-only text mode at each granularity.
ALSO_RUN_CLAIM_ONLY = False

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Config loaded. Output dir:", OUTPUT_DIR)