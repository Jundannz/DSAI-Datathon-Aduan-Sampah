import os
import re
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
import torch.nn as nn
from difflib import SequenceMatcher

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

warnings.filterwarnings("ignore")

# ── Konfigurasi Global ──────────────────────────────────────────────────────
CONFIG = {
    # Path — sesuaikan ke lokasi CSV kamu
    "csv_path"       : "/kaggle/input/datasets/jundansaifulhaq/dataset-merged-final-fix/dataset_merged_final.csv",
    "output_dir"     : "/kaggle/working/results_v3",
    "model_save_dir" : "/kaggle/working/model_final_v3",

    # Model
    "model_name"     : "indobenchmark/indobert-base-p1",
    "num_labels"     : 4,
    "max_length"     : 128,

    # Stratified split
    "test_size"      : 0.10,
    "val_size"       : 0.10,

    # Near-duplicate detection
    "dedup_threshold": 0.75,

    # Regularisasi
    "dropout"        : 0.30,
    "warmup_ratio"   : 0.10,

    # Hyperparameter search (Optuna)
    "n_trials"       : 5,
    "search_epochs"  : 3,    # epochs per trial — lebih pendek untuk efisiensi

    # Final training
    "num_epochs"     : 6,
    "batch_size"     : 16,

    # Reprodusibilitas
    "seed"           : 42,
}

# ── Label Mapping — 4 kelas termasuk Unknown ────────────────────────────────
# CSV menyimpan label dalam lowercase; normalisasi ke Title Case di Cell 3.
LABEL2ID = {"Rendah": 0, "Sedang": 1, "Tinggi": 2, "Unknown": 3}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# ── Seed & Device ────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device   : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Labels   : {LABEL2ID}")
print(f"Model    : {CONFIG['model_name']}")