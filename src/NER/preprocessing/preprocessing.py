def load_conll(filepath):
    """
    Membaca file CoNLL BIO. Toleran terhadap BOM dan CRLF.
    Mengembalikan list kalimat; setiap kalimat = list of (token, label).
    """
    sentences, current = [], []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current.append((parts[0], parts[-1]))
    if current:
        sentences.append(current)
    return sentences


all_sentences  = load_conll(CONFIG["data_path"])
all_label_flat = [lbl for sent in all_sentences for _, lbl in sent]
label_counts   = Counter(all_label_flat)
lengths        = [len(s) for s in all_sentences]
total_tokens   = sum(label_counts.values())

print(f"Total kalimat     : {len(all_sentences)}")
print(f"Total token       : {total_tokens:,}")
print(f"Rata-rata panjang : {sum(lengths)/len(lengths):.1f} token")
print(f"Panjang maks      : {max(lengths)} token")
print()
print("Distribusi label:")
for lbl, c in label_counts.most_common():
    bar = "█" * int(c / total_tokens * 50)
    print(f"  {lbl:<8}: {c:>6} ({c/total_tokens*100:5.1f}%) {bar}")

count_O = label_counts.get("O",     1)
count_B = label_counts.get("B-LOC", 1)
count_I = label_counts.get("I-LOC", 1)

CLASS_WEIGHTS_GLOBAL = torch.tensor([
    1.0,
    math.sqrt(count_O / count_B),
    math.sqrt(count_O / count_I),
], dtype=torch.float)

print("Class weights (global, akan di-update setelah split):")
print(f"  O      (idx=0): {CLASS_WEIGHTS_GLOBAL[0]:.4f}")
print(f"  B-LOC  (idx=1): {CLASS_WEIGHTS_GLOBAL[1]:.4f}")
print(f"  I-LOC  (idx=2): {CLASS_WEIGHTS_GLOBAL[2]:.4f}")

def has_loc(sentence):
    return any(lbl in ("B-LOC", "I-LOC") for _, lbl in sentence)

labels_for_split = [1 if has_loc(s) else 0 for s in all_sentences]

train_val_sents, test_sents, tv_lbl, _ = train_test_split(
    all_sentences, labels_for_split,
    test_size=CONFIG["test_size"],
    stratify=labels_for_split,
    random_state=CONFIG["seed"],
)

val_ratio = CONFIG["val_size"] / (1 - CONFIG["test_size"])
tv_strat  = [1 if has_loc(s) else 0 for s in train_val_sents]

train_sents, val_sents = train_test_split(
    train_val_sents,
    test_size=val_ratio,
    stratify=tv_strat,
    random_state=CONFIG["seed"],
)

print("Split:")
for name, split in [("Train", train_sents), ("Val", val_sents), ("Test", test_sents)]:
    n_loc = sum(1 for s in split if has_loc(s))
    print(f"  {name:<6}: {len(split):>5} kalimat | LOC: {n_loc}/{len(split)} ({n_loc/len(split)*100:.1f}%)")

# Update class weights dari training set saja
train_labels_flat = [lbl for sent in train_sents for _, lbl in sent]
train_cnt = Counter(train_labels_flat)
CLASS_WEIGHTS = torch.tensor([
    1.0,
    math.sqrt(train_cnt.get("O", 1) / train_cnt.get("B-LOC", 1)),
    math.sqrt(train_cnt.get("O", 1) / train_cnt.get("I-LOC", 1)),
], dtype=torch.float)
print(f"\nClass weights (dari train set): O={CLASS_WEIGHTS[0]:.3f}, B-LOC={CLASS_WEIGHTS[1]:.3f}, I-LOC={CLASS_WEIGHTS[2]:.3f}")

print(f"Memuat tokenizer: {CONFIG['model_name']}")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_name"],
    do_lower_case=True,   # FIX: normalisasi lowercase eksplisit
    use_fast=True,        # FIX: pastikan fast tokenizer (wajib untuk word_ids() akurat)
)
print(f"Tokenizer type : {type(tokenizer).__name__}")
print(f"Vocab size     : {tokenizer.vocab_size:,}")
print(f"Is fast?       : {tokenizer.is_fast}")
print()

# ── SANITY CHECK: Pastikan word_ids() bekerja benar ──────────────────────────
# Ini adalah verifikasi yang WAJIB dilakukan sebelum training.
# Jika word_ids() mengembalikan None atau semua None, alignment akan rusak
# dan seluruh training akan sia-sia (F1 = 0).

test_words  = ["jalan", "kaliurang", "km", "10"]
test_labels = ["B-LOC", "I-LOC", "I-LOC", "I-LOC"]

enc      = tokenizer(test_words, is_split_into_words=True, return_tensors=None)
word_ids = enc.word_ids()
subwords = tokenizer.convert_ids_to_tokens(enc["input_ids"])

print("── Sanity Check word_ids() ──────────────────────────────")
print(f"{'Subword':<18} {'word_id':<10} {'Expected label'}")
n_valid = 0
n_invalid = 0
for sw, wid in zip(subwords, word_ids):
    if wid is not None:
        lbl = test_labels[wid] if wid < len(test_labels) else "?"
        n_valid += 1
    else:
        lbl = "SPECIAL/IGNORE"
        n_invalid += 1
    print(f"  {sw:<16} {str(wid):<10} {lbl}")

# Pastikan ada word_id yang valid (bukan semua None)
n_loc_ids = sum(1 for wid in word_ids if wid is not None)
assert n_loc_ids > 0, "FATAL: word_ids() mengembalikan semua None! Tokenizer tidak kompatibel."
print()
print(f"✓ word_ids() bekerja: {n_valid} valid tokens, {n_invalid} special tokens")
print("✓ Alignment label akan bekerja benar")
print()

# ── Demo alignment untuk berbagai kasus casing ────────────────────────────────
print("── Demo Alignment (kasus casing & angka) ───────────────")


def align_labels_with_word_ids(word_ids: list, original_labels: list, label2id: dict) -> list:
    """
    Memetakan label BIO dari level kata ke level sub-word token.

    Rules:
    - Token spesial (word_id is None)   → -100  (diabaikan di CrossEntropy)
    - Sub-word PERTAMA dari kata baru   → label asli kata tersebut
    - Sub-word LANJUTAN (##xxx)         → -100  (diabaikan di CrossEntropy)
    """
    aligned, prev_word_id = [], None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != prev_word_id:
            aligned.append(label2id[original_labels[word_id]])
        else:
            aligned.append(-100)
        prev_word_id = word_id
    return aligned


demo_cases = [
    (["Jalan", "Kaliurang", "KM", "10"],             ["B-LOC", "I-LOC", "I-LOC", "I-LOC"]),
    (["jalan", "Kaliurang", "km", "10"],             ["B-LOC", "I-LOC", "I-LOC", "I-LOC"]),
    (["jalan", "kaliurang", "km", "10"],             ["B-LOC", "I-LOC", "I-LOC", "I-LOC"]),
    (["jalan", "Sleman", "Ring", "Road", "Selatan"], ["B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC"]),
    (["jalan", "Kenanga", "RT", "02", "RW", "01"],   ["B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC"]),
    (["SLB", "N", "2", "Bantul"],                    ["B-LOC", "I-LOC", "I-LOC", "I-LOC"]),
]

all_ok = True
for words, labels in demo_cases:
    enc     = tokenizer(words, is_split_into_words=True,
                        truncation=True, max_length=CONFIG["max_length"])
    wids    = enc.word_ids()
    aligned = align_labels_with_word_ids(wids, labels, CONFIG["label2id"])
    sw      = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    # Hitung berapa label non-O yang terdeteksi (harus sama dengan jumlah kata LOC)
    n_loc_aligned = sum(1 for a in aligned if a in (1, 2))
    n_loc_words   = sum(1 for l in labels if l != "O")

    status = "✓" if n_loc_aligned == n_loc_words else "✗ MISMATCH"
    if n_loc_aligned != n_loc_words:
        all_ok = False
    print(f"  {status} '{' '.join(words)}' → {n_loc_aligned}/{n_loc_words} LOC tokens aligned")

print()
if all_ok:
    print("✓ Semua kasus alignment benar! Training siap dimulai.")
else:
    print("✗ Ada alignment mismatch! Cek tokenizer. JANGAN lanjutkan training.")
    
def sentences_to_hf_dataset(sentences):
    """Konversi list kalimat ke HF Dataset siap pakai Trainer."""
    tokens_list, labels_list = [], []
    for sent in sentences:
        toks = [t for t, _ in sent]
        lbls = [l for _, l in sent]
        enc  = tokenizer(
            toks,
            is_split_into_words=True,
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=False,
            return_tensors=None,
        )
        aligned = align_labels_with_word_ids(enc.word_ids(), lbls, CONFIG["label2id"])
        tokens_list.append(enc["input_ids"])
        labels_list.append(aligned)

    return Dataset.from_dict({
        "input_ids"      : tokens_list,
        "attention_mask" : [[1] * len(ids) for ids in tokens_list],
        "labels"         : labels_list,
    })


print("Tokenisasi training set...")
train_tok = sentences_to_hf_dataset(train_sents)
print("Tokenisasi validation set...")
val_tok   = sentences_to_hf_dataset(val_sents)
print("Tokenisasi test set...")
test_tok  = sentences_to_hf_dataset(test_sents)

# Verifikasi: hitung B-LOC/I-LOC yang berhasil di-align di train set
n_bloc = sum(1 for seq in train_tok["labels"] for l in seq if l == 1)
n_iloc = sum(1 for seq in train_tok["labels"] for l in seq if l == 2)
n_o    = sum(1 for seq in train_tok["labels"] for l in seq if l == 0)
n_ign  = sum(1 for seq in train_tok["labels"] for l in seq if l == -100)

print(f"\nUkuran dataset (post-tokenisasi):")
print(f"  Train : {len(train_tok):>5}")
print(f"  Val   : {len(val_tok):>5}")
print(f"  Test  : {len(test_tok):>5}")
print(f"\nLabel distribution di train (setelah alignment):")
print(f"  O      : {n_o:>6}")
print(f"  B-LOC  : {n_bloc:>6}")
print(f"  I-LOC  : {n_iloc:>6}")
print(f"  Ignored: {n_ign:>6}  (sub-word lanjutan + [CLS]/[SEP])")
assert n_bloc > 0, "FATAL: Tidak ada B-LOC di training set setelah alignment! Cek tokenizer."
assert n_iloc > 0, "FATAL: Tidak ada I-LOC di training set setelah alignment! Cek tokenizer."
print(f"\n✓ B-LOC dan I-LOC berhasil di-align ke training labels.")

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)
print("DataCollator siap (dynamic padding, label_pad=-100)")
