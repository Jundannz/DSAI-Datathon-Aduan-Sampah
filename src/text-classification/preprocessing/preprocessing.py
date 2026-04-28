df = pd.read_csv(CONFIG["csv_path"])

print(f"Total baris raw     : {len(df)}")
print(f"Kolom               : {df.columns.tolist()}")
print(f"\nMissing values per kolom:\n{df.isnull().sum().to_string()}")
print(f"\nContoh data:")
display(df.head(3))

# ── Normalisasi label: lowercase -> Title Case ───────────────────────────────
# CSV menyimpan 'tinggi', 'rendah', 'sedang', 'unknown' (lowercase).
# LABEL2ID menggunakan Title Case → perlu .str.title() sebelum mapping.
df["label_prioritas"] = df["label_prioritas"].str.strip().str.title()

# Hapus baris dengan label di luar domain yang dikenal
valid_labels = set(LABEL2ID.keys())
invalid_mask = ~df["label_prioritas"].isin(valid_labels)
if invalid_mask.sum() > 0:
    print(f"\n{invalid_mask.sum()} baris dengan label tidak dikenal -> dihapus")
    df = df[~invalid_mask].reset_index(drop=True)

print(f"\nTotal baris setelah validasi label : {len(df)}")

# ── Visualisasi Distribusi Label ─────────────────────────────────────────────
label_counts = df["label_prioritas"].value_counts()
PALETTE = {"Tinggi": "#e74c3c", "Sedang": "#f39c12", "Rendah": "#2ecc71", "Unknown": "#7f8c8d"}

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Bar chart dengan annotation jumlah & persentase
colors = [PALETTE.get(l, "#bdc3c7") for l in label_counts.index]
bars   = axes[0].bar(label_counts.index, label_counts.values, color=colors,
                     edgecolor="white", linewidth=1.5, zorder=3)
axes[0].set_title("Distribusi Label (Raw Dataset)", fontsize=13, fontweight="bold")
axes[0].set_ylim(0, label_counts.max() * 1.30)
axes[0].set_ylabel("Jumlah Sampel", fontsize=11)
axes[0].grid(axis="y", alpha=0.3, zorder=0)
for bar, val in zip(bars, label_counts.values):
    pct = val / len(df) * 100
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 4,
                 f"{val}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

# Pie chart
pie_colors = [PALETTE.get(l, "#bdc3c7") for l in label_counts.index]
wedges, texts, autotexts = axes[1].pie(
    label_counts.values,
    labels=label_counts.index,
    colors=pie_colors,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 11},
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axes[1].set_title("Proporsi Label (%)", fontsize=13, fontweight="bold")

plt.suptitle("EDA — Distribusi Kelas Dataset", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# ── Analisis Imbalance ───────────────────────────────────────────────────────
min_c, max_c    = label_counts.min(), label_counts.max()
imbalance_ratio = max_c / min_c

print(f"\nDistribusi label    : {dict(label_counts)}")
print(f"Imbalance ratio     : {imbalance_ratio:.2f}x")

if imbalance_ratio < 1.2:
    print("Dataset SEIMBANG — class weights tetap dihitung sebagai safeguard.")
elif imbalance_ratio < 1.5:
    print("Imbalance MINOR — class weights akan diaktifkan saat training.")
else:
    print("Imbalance SIGNIFIKAN — class weights wajib digunakan.")

# Near-duplicate terjadi karena data LLM-generated menggunakan template yang sama
# dengan hanya nama lokasi yang diganti. Pasangan seperti ini menyebabkan data leakage
# jika template yang sama masuk ke train sekaligus ke val/test.
#
# Strategi: bandingkan 200 karakter pertama teks (bagian paling berulang).
# Jika similarity >= threshold, hapus instance yang lebih akhir (keep first).

THRESHOLD = CONFIG["dedup_threshold"]

print(f"Scanning near-duplicates (threshold >= {THRESHOLD})...")
print(f"Perbandingan berbasis 200 karakter pertama teks\n")

texts   = df["teks_laporan"].tolist()
ids     = df["id_laporan"].tolist()
n_raw   = len(texts)
to_drop = set()
dup_log = []

for i in range(n_raw):
    for j in range(i + 1, n_raw):
        if j in to_drop:
            continue
        ratio = SequenceMatcher(None, texts[i][:200], texts[j][:200]).ratio()
        if ratio >= THRESHOLD:
            to_drop.add(j)
            dup_log.append({
                "id_a"  : ids[i],
                "id_b"  : ids[j],
                "sim"   : round(ratio, 3),
                "text_a": texts[i][:85] + "...",
                "text_b": texts[j][:85] + "...",
            })

if dup_log:
    df_log = pd.DataFrame(dup_log).sort_values("sim", ascending=False)
    print(f"Ditemukan   : {len(dup_log)} pasangan near-duplicate")
    print(f"Akan dihapus: {len(to_drop)} baris (keep first, drop later)\n")

    print("TOP-10 PASANGAN PALING MIRIP")
    print("-" * 60)
    for _, row in df_log.head(10).iterrows():
        print(f"[{row['id_a']} vs {row['id_b']}]  sim={row['sim']}")
        print(f"  A: {row['text_a']}")
        print(f"  B: {row['text_b']}")
        print()

    # Visualisasi distribusi similarity score
    sims = df_log["sim"].tolist()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(sims, bins=20, color="#3498db", edgecolor="white", zorder=3)
    ax.axvline(THRESHOLD, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Threshold = {THRESHOLD}")
    ax.set_xlabel("Similarity Score", fontsize=11)
    ax.set_ylabel("Jumlah Pasangan", fontsize=11)
    ax.set_title("Distribusi Similarity Near-Duplicate Pairs", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, zorder=0)
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Tidak ada near-duplicate ditemukan.")

# Terapkan filter
keep_mask = [i not in to_drop for i in range(n_raw)]
df_clean  = df[keep_mask].reset_index(drop=True)

print(f"\nSebelum dedup : {n_raw} baris")
print(f"Setelah dedup : {len(df_clean)} baris  (-{n_raw - len(df_clean)})")
print(f"\nDistribusi label setelah dedup:")
print(df_clean["label_prioritas"].value_counts().to_string())

def clean_text(text: str) -> str:
    """
    Preprocessing minimalis — sengaja tidak agresif.

    Pendekatan ini mempertahankan slang, singkatan, dan bahasa kasual karena:
      1. Dataset berisi laporan warga dengan bahasa informal/gaul
      2. IndoBERT menangani OOV melalui subword tokenisasi (WordPiece)
      3. Stemming dan stop-word removal merusak sinyal semantik kalimat

    Yang dibersihkan:
      - Karakter kontrol Unicode (\x00-\x1f, \x7f-\x9f)
      - Baris baru dan tab -> diganti spasi
      - Simbol non-relevan di luar alfanumerik dan tanda baca umum
      - Spasi ganda berlebih
    """
    if not isinstance(text, str):
        return ""
    # Hapus karakter kontrol
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Normalisasi whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Pertahankan alfanumerik, tanda baca umum, dan Unicode (aksara Jawa, dll.)
    text = re.sub(r"[^\w\s.,!?;:\-\'\"()/]", " ", text, flags=re.UNICODE)
    text = re.sub(r" {2,}", " ", text).strip()
    return text.lower()


# Terapkan ke seluruh dataset bersih
df_clean = df_clean.copy()
df_clean["teks_bersih"] = df_clean["teks_laporan"].apply(clean_text)
df_clean["label"]       = df_clean["label_prioritas"].map(LABEL2ID)

# Sanity check: tidak boleh ada NaN setelah mapping
n_nan = df_clean["label"].isna().sum()
assert n_nan == 0, f"{n_nan} label NaN terdeteksi — periksa LABEL2ID vs nilai CSV"
print("Label mapping valid — tidak ada NaN\n")

# Preview contoh per kelas
print("CONTOH TEKS PER KELAS (setelah clean)")
print("=" * 65)
for label_name in LABEL2ID.keys():
    sample_row = df_clean[df_clean["label_prioritas"] == label_name].iloc[0]
    print(f"[{label_name}]")
    print(f"  Original : {sample_row['teks_laporan'][:110]}...")
    print(f"  Cleaned  : {sample_row['teks_bersih'][:110]}...")
    print()

# Stratified split memastikan proporsi setiap kelas (Rendah, Sedang, Tinggi, Unknown)
# sama di split train, val, dan test.
# Rasio 80/10/10 dipilih untuk memaksimalkan data training.

TEST_SIZE = CONFIG["test_size"]   # 0.10
VAL_SIZE  = CONFIG["val_size"]    # 0.10

# Step 1: pisahkan test set (10%)
df_trainval, df_test = train_test_split(
    df_clean,
    test_size=TEST_SIZE,
    stratify=df_clean["label_prioritas"],
    random_state=CONFIG["seed"],
)

# Step 2: dari sisa trainval, pisahkan val
# val_ratio = 0.10 / 0.90 ≈ 0.111 agar val = 10% dari total
val_ratio = VAL_SIZE / (1 - TEST_SIZE)
df_train, df_val = train_test_split(
    df_trainval,
    test_size=val_ratio,
    stratify=df_trainval["label_prioritas"],
    random_state=CONFIG["seed"],
)

print(f"Stratified Split (total: {len(df_clean)} baris)")
print(f"  Train : {len(df_train):>5} ({len(df_train)/len(df_clean)*100:.1f}%)")
print(f"  Val   : {len(df_val):>5} ({len(df_val)/len(df_clean)*100:.1f}%)")
print(f"  Test  : {len(df_test):>5} ({len(df_test)/len(df_clean)*100:.1f}%)")

# ── Visualisasi distribusi per split ────────────────────────────────────────
splits = [("Train", df_train), ("Validation", df_val), ("Test", df_test)]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, df_split) in zip(axes, splits):
    counts = df_split["label_prioritas"].value_counts()
    colors = [PALETTE.get(l, "#bdc3c7") for l in counts.index]
    bars   = ax.bar(counts.index, counts.values, color=colors,
                    edgecolor="white", linewidth=1.3, zorder=3)
    ax.set_title(f"{title}  (n={len(df_split)})", fontsize=12, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.35)
    ax.set_ylabel("Jumlah Sampel", fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(val), ha="center", fontsize=11, fontweight="bold")

plt.suptitle("Distribusi Label per Split (Stratified 80/10/10)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# Detail per kelas per split
for name, df_split in splits:
    print(f"\n{name}:")
    print(df_split["label_prioritas"].value_counts().to_string())

# Class weights memberikan bobot lebih besar pada kelas yang lebih sedikit,
# sehingga model tidak bias ke kelas mayoritas saat training.
# Dihitung hanya dari train set untuk menghindari leakage dari val/test.

train_labels_int = df_train["label"].values

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(sorted(LABEL2ID.values())),
    y=train_labels_int,
)

# Simpan sebagai tensor PyTorch di device yang sesuai
CLASS_WEIGHTS_TENSOR = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

print("Class Weights (digunakan di CustomTrainer -> CrossEntropyLoss):")
print("-" * 50)
print(f"{'Label':<12} {'ID':<5} {'Weight':<10} Visualisasi")
print("-" * 50)
for label_id in sorted(ID2LABEL.keys()):
    label_name = ID2LABEL[label_id]
    weight     = class_weights[label_id]
    bar        = "#" * int(weight * 25)
    print(f"{label_name:<12} {label_id:<5} {weight:<10.4f} {bar}")

n_per_class = pd.Series(train_labels_int).value_counts().sort_index()
print("\nJumlah sampel per kelas di train set:")
for cls_id, n in n_per_class.items():
    print(f"  {ID2LABEL[cls_id]:<10} : {n} sampel")

print(f"\nCatatan: weight > 1.0 = kelas minoritas (gradient diperbesar)")
print(f"         weight < 1.0 = kelas mayoritas (gradient diperkecil)")
print(f"\nTensor: shape={CLASS_WEIGHTS_TENSOR.shape} | device={CLASS_WEIGHTS_TENSOR.device}")

print(f"Memuat tokenizer: {CONFIG['model_name']}")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
print(f"Vocab size: {tokenizer.vocab_size:,}")

def tokenize_fn(examples: dict) -> dict:
    """
    Tokenisasi teks menggunakan tokenizer IndoBERT.
    Menggunakan truncation=True tanpa padding di sini karena padding
    ditangani secara dinamis oleh DataCollatorWithPadding (lebih efisien).
    """
    return tokenizer(
        examples["teks_bersih"],
        truncation=True,
        max_length=CONFIG["max_length"],
    )


def df_to_hf(df_split: pd.DataFrame, split_name: str = "") -> Dataset:
    """
    Konversi DataFrame ke HuggingFace Dataset dan lakukan tokenisasi
    secara batched untuk efisiensi memori.

    Kolom 'labels' (bukan 'label') digunakan karena Trainer HuggingFace
    secara default mencari kolom bernama 'labels' untuk loss computation.
    """
    ds = Dataset.from_dict({
        "teks_bersih": df_split["teks_bersih"].tolist(),
        "labels"     : df_split["label"].astype(int).tolist(),
    })
    desc = f"Tokenisasi [{split_name}]" if split_name else "Tokenisasi"
    ds   = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=64,
        desc=desc,
    )
    # Hapus kolom teks mentah — tidak dibutuhkan saat training
    ds = ds.remove_columns(["teks_bersih"])
    return ds


# DataCollatorWithPadding menerapkan dynamic padding per batch
# lebih hemat memori dibandingkan padding ke max_length secara statis
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("\nMemproses split ke HuggingFace Dataset...")
train_dataset = df_to_hf(df_train, "train")
val_dataset   = df_to_hf(df_val,   "val")
test_dataset  = df_to_hf(df_test,  "test")

print("\nDataset HuggingFace siap:")
print(f"  Train : {len(train_dataset):>5} sampel")
print(f"  Val   : {len(val_dataset):>5} sampel")
print(f"  Test  : {len(test_dataset):>5} sampel")
print(f"  Features: {list(train_dataset.features.keys())}")

# Cek distribusi panjang token pada train set
token_lengths = [len(x) for x in train_dataset["input_ids"]]
print(f"\nDistribusi panjang token (train set):")
print(f"  Min    : {min(token_lengths)}")
print(f"  Max    : {max(token_lengths)}")
print(f"  Mean   : {np.mean(token_lengths):.1f}")
print(f"  Median : {np.median(token_lengths):.1f}")
print(f"  P95    : {np.percentile(token_lengths, 95):.1f}")
print(f"  Max length config: {CONFIG['max_length']}")
