import os, json, random, warnings, math
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from datasets import Dataset
from seqeval.metrics import (
    f1_score        as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score    as seqeval_recall,
    classification_report as seqeval_report,
)

warnings.filterwarnings("ignore")

CONFIG = {
    # ── Path ──────────────────────────────────────────────────────────────────
    "data_path"  : "/kaggle/input/datasets/jundansaifulhaq/dataset-ner-rev2/dataset_ner_cleaned.txt",
    "output_dir" : "/kaggle/working/ner_bert_uncased_v2",

    # ── Model ─────────────────────────────────────────────────────────────────
    "model_name" : "indobenchmark/indobert-base-p2",
    "max_length" : 128,      # BERT WordPiece lebih efisien; 128 aman untuk semua kalimat

    # ── Label ─────────────────────────────────────────────────────────────────
    "label2id"   : {"O": 0, "B-LOC": 1, "I-LOC": 2},

    # ── Split ─────────────────────────────────────────────────────────────────
    "test_size"  : 0.10,
    "val_size"   : 0.10,

    # ── Training ──────────────────────────────────────────────────────────────
    "num_epochs"          : 20,    # lebih banyak epoch karena dataset kecil
    "batch_size"          : 16,
    "learning_rate"       : 2e-5,
    "warmup_ratio"        : 0.10,
    "weight_decay"        : 0.01,
    "max_grad_norm"       : 1.0,
    "dropout"             : 0.15,
    "early_stop_patience" : 5,

    "seed" : 42,
}

ID2LABEL   = {v: k for k, v in CONFIG["label2id"].items()}
NUM_LABELS = len(CONFIG["label2id"])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device    : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Model     : {CONFIG['model_name']}")
print(f"Labels    : {CONFIG['label2id']}")