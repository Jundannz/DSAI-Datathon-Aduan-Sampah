best_model = final_trainer.model
best_model.eval()

# ── Prediksi test set ─────────────────────────────────────────────────────────
test_output = final_trainer.predict(test_dataset)
logits      = test_output.predictions
true_labels = test_output.label_ids
pred_ids    = np.argmax(logits, axis=-1)

class_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
true_names  = [ID2LABEL[i] for i in true_labels]
pred_names  = [ID2LABEL[i] for i in pred_ids]

# ── Hitung metrik final ───────────────────────────────────────────────────────
f1_macro   = f1_score(true_labels, pred_ids, average="macro",      zero_division=0)
f1_per_cls = f1_score(true_labels, pred_ids, average=None,
                      labels=sorted(ID2LABEL.keys()),               zero_division=0)
accuracy   = accuracy_score(true_labels, pred_ids)
precision  = precision_score(true_labels, pred_ids, average="macro", zero_division=0)
recall     = recall_score(true_labels, pred_ids, average="macro",    zero_division=0)

print("=" * 60)
print("  EVALUASI FINAL — TEST SET")
print("=" * 60)
print(f"  Macro F1-Score       : {f1_macro:.4f}  ({f1_macro*100:.2f}%)")
print(f"  Accuracy             : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Macro Precision      : {precision:.4f}")
print(f"  Macro Recall         : {recall:.4f}")
print()
print("  F1-Score per Kelas:")
bar_max = 35
for i, name in enumerate(class_names):
    score = f1_per_cls[i]
    bar   = "#" * int(score * bar_max)
    print(f"    {name:<10} : {score:.4f}  [{bar:<{bar_max}}]")

print(f"\n  Classification Report:\n")
print(classification_report(true_names, pred_names,
                             target_names=class_names, zero_division=0))

# ── Confusion Matrix (Count + Percentage) ────────────────────────────────────
cm     = confusion_matrix(true_labels, pred_ids, labels=sorted(ID2LABEL.keys()))
cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix — jumlah absolut
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", linewidths=0.6,
    xticklabels=class_names, yticklabels=class_names, ax=axes[0],
    annot_kws={"size": 13, "fontweight": "bold"},
)
axes[0].set_title(
    f"Confusion Matrix (Jumlah Absolut)\nF1-macro = {f1_macro:.4f}  |  Accuracy = {accuracy:.4f}",
    fontsize=12, fontweight="bold"
)
axes[0].set_xlabel("Predicted", fontsize=11)
axes[0].set_ylabel("Actual", fontsize=11)
axes[0].tick_params(axis="x", rotation=25)
axes[0].tick_params(axis="y", rotation=0)

# Confusion Matrix — persentase per kelas actual (normalisasi per baris)
annot_pct = np.array([
    [f"{v:.1f}%" for v in row] for row in cm_pct
])
sns.heatmap(
    cm_pct, annot=annot_pct, fmt="", cmap="Oranges", linewidths=0.6,
    xticklabels=class_names, yticklabels=class_names, ax=axes[1],
    annot_kws={"size": 12},
    vmin=0, vmax=100,
)
axes[1].set_title(
    "Confusion Matrix (% per Kelas Actual)\nNormalisasi per baris",
    fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Predicted", fontsize=11)
axes[1].set_ylabel("Actual", fontsize=11)
axes[1].tick_params(axis="x", rotation=25)
axes[1].tick_params(axis="y", rotation=0)

plt.suptitle("Evaluasi Akhir — Test Set", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# ── Verdict ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if f1_macro >= 0.80:
    print("  PERFORMA SANGAT BAIK — Model siap untuk deployment")
elif f1_macro >= 0.65:
    print("  PERFORMA BAIK — Model layak pakai, masih dapat ditingkatkan")
elif f1_macro >= 0.50:
    print("  PERFORMA SEDANG — Perlu iterasi lebih lanjut")
else:
    print("  PERFORMA RENDAH — Periksa data, label, atau strategi training")
print("=" * 60)

SAVE_DIR = CONFIG["model_save_dir"]
os.makedirs(SAVE_DIR, exist_ok=True)

# Pindahkan ke CPU sebelum menyimpan agar kompatibel lintas lingkungan
best_model.eval().to("cpu")
best_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Metadata lengkap termasuk metrik evaluasi dan best hyperparams
metadata = {
    "model_base"        : CONFIG["model_name"],
    "task"              : "text-classification",
    "num_labels"        : CONFIG["num_labels"],
    "label2id"          : LABEL2ID,
    "id2label"          : {str(k): v for k, v in ID2LABEL.items()},
    "max_length"        : CONFIG["max_length"],
    "quantization"      : "none (FP32)",
    "dataset_info": {
        "total_after_dedup" : len(df_clean),
        "dedup_removed"     : n_raw - len(df_clean),
        "split"             : "80/10/10 stratified",
        "train_size"        : len(df_train),
        "val_size"          : len(df_val),
        "test_size"         : len(df_test),
    },
    "evaluation_test": {
        "f1_macro"          : round(f1_macro, 4),
        "accuracy"          : round(accuracy, 4),
        "precision_macro"   : round(precision, 4),
        "recall_macro"      : round(recall, 4),
        "f1_per_class"      : {class_names[i]: round(float(f1_per_cls[i]), 4)
                               for i in range(len(class_names))},
    },
    "best_hyperparameters": best_run.hyperparameters,
    "optuna_n_trials"     : CONFIG["n_trials"],
}

with open(os.path.join(SAVE_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Semua file tersimpan ke: {SAVE_DIR}")
print("  pytorch_model.bin       — bobot model FP32")
print("  config.json             — konfigurasi arsitektur")
print("  tokenizer_config.json   — konfigurasi tokenizer")
print("  vocab.txt               — vocabulari IndoBERT")
print("  model_metadata.json     — metadata & metrik evaluasi")

print("\nRingkasan metadata:")
eval_info = metadata["evaluation_test"]
print(f"  F1-macro (test) : {eval_info['f1_macro']}")
print(f"  Accuracy (test) : {eval_info['accuracy']}")
print(f"  F1 per kelas    : {eval_info['f1_per_class']}")
print(f"  Best LR         : {best_run.hyperparameters.get('learning_rate', 'N/A'):.2e}")
print(f"  Best WD         : {best_run.hyperparameters.get('weight_decay', 'N/A'):.4f}")

def predict_priority(text: str) -> dict:
    """
    Prediksi prioritas laporan dari teks bebas.
    Mengembalikan label, confidence, dan skor per kelas.
    """
    best_model.eval()
    cleaned = clean_text(text)
    inputs  = tokenizer(
        cleaned,
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length",
        return_tensors="pt",
    ).to("cpu")
    with torch.no_grad():
        outputs = best_model.cpu()(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_id = int(np.argmax(probs))
    return {
        "label"      : ID2LABEL[pred_id],
        "confidence" : round(float(probs[pred_id]), 4),
        "scores"     : {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def run_inference(text: str) -> None:
    """
    Tampilkan hasil prediksi satu teks secara terformat.
    """
    result  = predict_priority(text)
    label   = result["label"]
    conf    = result["confidence"]
    scores  = result["scores"]
    bar_len = 30

    print("=" * 65)
    print(f"  Input    : {text[:70]}{'...' if len(text) > 70 else ''}")
    print(f"  Prediksi : {label}  (confidence {conf:.4f} / {conf*100:.1f}%)")
    print("  Skor per kelas:")
    for lbl in class_names:
        sc     = scores[lbl]
        bar    = "#" * int(sc * bar_len)
        marker = " <-- PREDIKSI" if lbl == label else ""
        print(f"    {lbl:<10} : {sc:.4f}  [{bar:<{bar_len}}]{marker}")
    print()


# ── Sanity Check ─────────────────────────────────────────────────────────────
sanity_texts = [
    # Ekspektasi: Tinggi
    "Woi darurat! di depan SDN Condong Catur belatung udah keluar kemana-mana, anak-anak pada muntah",
    "Njir limbah pabrik dibuang ke Kali Code lagi, airnya item dan bau banget, ikan pada mati",
    # Ekspektasi: Sedang
    "Eh min, sampah di depan Pasar Giwangan udah numpuk 5 hari, baunya mulai ga enak nih",
    "Pak ini TPS di jalan Kaliurang sudah penuh seminggu, tolong segera diambil",
    # Ekspektasi: Rendah
    "Permisi mau nanya, jadwal truk sampah di Jalan Kaliurang KM 8 itu hari apa aja ya?",
    "maaf pak mumpung masih belum terlalu banyak apakah bisa diambil sampahnya",
    # Ekspektasi: Unknown / Ambigu
    "Apresiasi bgt buat petugas kebersihan Mantrijeron, tiap pagi udah rajin beresin sampah",
    "Permisi pak, fyi aja gw baru pindah ke area Pogung, mau buang kasur lipat yg udah rusak",
    # Custom:
    "Pak ini ada satu sampah yang ketinggalan, tadi lupa belum keangkut sama petugas, tapi kalau mau diambil di next pengambilan boleh sih",
    "Izin lapor min, tadi pagi sekitar jam 6 ada tabrakan beruntun di jalan jogja solo, dimohon untuk orang buat senantiasa hati hati nggih"
]

print("SANITY CHECK INFERENSI")
print(f"Model: {CONFIG['model_name']} — 4 Kelas: {list(LABEL2ID.keys())}\n")
for text in sanity_texts:
    run_inference(text)
