"""
run_all.py

Complete, ordered, end-to-end run of every experiment built in this
project. Paste this into a Kaggle notebook (as one script cell, or split
at the "# ===== SECTION" markers into separate cells) AFTER pasting in
every class file this script imports from:

    liar_preprocessor.py           liar2_preprocessor.py
    baseline_models.py             multifc_preprocessor.py
    temporal_splitter.py           da_minilm_model.py
    da_minilm_experiment_runner.py multifc_experiment.py
    shuffle_control.py             drift_analysis.py

Requires internet enabled on the Kaggle notebook (Settings -> Internet ->
On) for: DA-MiniLM-RWAC/MiniLM-LogReg-RW MiniLM embeddings, DistilBERT
fine-tuning, and MultiFC MiniLM embeddings. A GPU accelerator is strongly
recommended for the MLP/DistilBERT sections.

Update the four path variables in SECTION 0 to match your actual Kaggle
dataset paths before running.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from pathlib import Path

