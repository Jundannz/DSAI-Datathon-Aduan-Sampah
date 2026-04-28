print("Mengevaluasi model terbaik pada test set...\n")
test_metrics = trainer.evaluate(eval_dataset=test_tok)

print("=" * 60)
print("HASIL EVALUASI — TEST SET")
print("=" * 60)
for k, v in test_metrics.items():
    print(f"  {k:<35} : {v}")
print("=" * 60)

raw_pred  = trainer.predict(test_tok)
pred_ids  = np.argmax(raw_pred.predictions, axis=-1)
label_ids = raw_pred.label_ids

true_labels_report, true_preds_report = [], []
for pred_seq, label_seq in zip(pred_ids, label_ids):
    tl, tp = [], []
    for p, l in zip(pred_seq, label_seq):
        if l == -100: continue
        tl.append(ID2LABEL[l])
        tp.append(ID2LABEL[p])
    true_labels_report.append(tl)
    true_preds_report.append(tp)

print("\nClassification Report (seqeval, span-level):")
print(seqeval_report(true_labels_report, true_preds_report, zero_division=0))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

flat_preds = [p for seq in true_preds_report for p in seq]
flat_trues = [t for seq in true_labels_report for t in seq]

label_names = ["O", "B-LOC", "I-LOC"]
cm = confusion_matrix(flat_trues, flat_preds, labels=label_names)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — Test Set (Token Level)", fontweight="bold")
plt.tight_layout(); plt.show()

print("\n--- Contoh Kalimat dengan Kesalahan ---")
shown = 0
for sent, preds, trues in zip(test_sents, true_preds_report, true_labels_report):
    if preds != trues and shown < 5:
        tokens = [t for t, _ in sent[:len(trues)]]
        print(f"  Tokens : {tokens}")
        print(f"  True   : {trues}")
        print(f"  Pred   : {preds}\n")
        shown += 1

from transformers import pipeline as hf_pipeline

ner_pipeline = hf_pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0 if DEVICE.type == "cuda" else -1,
)


def run_ner_inference(text: str):
    results   = ner_pipeline(text)
    loc_spans = [r for r in results if r["entity_group"] == "LOC"]
    print(f"  Input : {text}")
    if loc_spans:
        locs = [f"'{r['word']}' ({r['score']:.2f})" for r in loc_spans]
        print(f"  LOC   : {', '.join(locs)}")
    else:
        print("  LOC   : (tidak terdeteksi)")
    print()


print("=" * 70)
print("SANITY CHECK — Model harus detect LOC untuk SEMUA casing di bawah ini")
print("=" * 70)
print()

test_cases = [
    # ── Kasus casing (ini yang bermasalah di notebook sebelumnya) ──────────
    ("Mau nanya jadwal truk sampah di Jalan Kaliurang KM 8 hari apa saja",
     "HARAPAN: 'Jalan Kaliurang KM 8'"),
    ("mau nanya jadwal truk sampah di jalan kaliurang km 8 hari apa saja",
     "HARAPAN: 'jalan kaliurang km 8'  [all lowercase]"),
    ("MINTA INFO JADWAL SAMPAH DI JALAN KALIURANG KM 8",
     "HARAPAN: 'JALAN KALIURANG KM 8'  [all uppercase]"),

    # ── Kasus boundary angka & singkatan ─────────────────────────────────
    ("Sampah menumpuk di jalan kaliurang km 10 sudah tiga hari tidak diangkut",
     "HARAPAN: 'jalan kaliurang km 10'"),
    ("di jalan Kenanga RT 02 RW 01 Umbulharjo ada tumpukan sampah besar",
     "HARAPAN: 'jalan Kenanga RT 02 RW 01 Umbulharjo'"),
    ("gawat banget ini jl Magelang KM 7 Sleman dekat perumahan",
     "HARAPAN: 'jl Magelang KM 7 Sleman'"),

    # ── Kasus umum ──────────────────────────────────────────────────────
    ("Woi darurat tumpukan sampah di depan SDN Condong Catur sudah meluber",
     "HARAPAN: 'SDN Condong Catur'"),
    ("Pak tolong cek TPS di belakang Pasar Demangan Gondokusuman bau banget",
     "HARAPAN: 'Pasar Demangan Gondokusuman'"),
    ("Air lindi dari TPS Piyungan mengalir ke Kali Gajahwong dan Kali Code",
     "HARAPAN: 'TPS Piyungan', 'Kali Gajahwong', 'Kali Code'"),
    ("Tidak ada sampah di sini",
     "HARAPAN: (tidak terdeteksi)"),
]

for text, harapan in test_cases:
    print(f"  [{harapan}]")
    run_ner_inference(text)

SAVE_DIR = CONFIG["output_dir"]
os.makedirs(SAVE_DIR, exist_ok=True)

trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model + tokenizer → {SAVE_DIR}/")

metadata = {
    "task"              : "named-entity-recognition",
    "language"          : "id",
    "base_model"        : CONFIG["model_name"],
    "architecture"      : "BERT (indobert-base-p2) + WeightedLossTrainer",
    "entities"          : ["LOC"],
    "label2id"          : CONFIG["label2id"],
    "id2label"          : {str(k): v for k, v in ID2LABEL.items()},
    "max_length"        : CONFIG["max_length"],
    "do_lower_case"     : True,
    "tokenizer_type"    : type(tokenizer).__name__,
    "train_sentences"   : len(train_sents),
    "val_sentences"     : len(val_sents),
    "test_sentences"    : len(test_sents),
    "split"             : "80/10/10 stratified",
    "loss_function"     : "WeightedCrossEntropyLoss",
    "class_weights"     : {
        "O"     : float(CLASS_WEIGHTS[0]),
        "B-LOC" : float(CLASS_WEIGHTS[1]),
        "I-LOC" : float(CLASS_WEIGHTS[2]),
    },
    "learning_rate"     : CONFIG["learning_rate"],
    "batch_size"        : CONFIG["batch_size"],
    "fix_alignment"     : "Switched ALBERT→BERT WordPiece for reliable word_ids()",
    "fix_class_imbalance": "WeightedLossTrainer with sqrt inverse freq weights",
    "fix_casing"        : "do_lower_case=True, indobert-base-p2 is uncased",
    "test_f1_LOC"       : test_metrics.get("eval_f1_LOC", None),
}
with open(os.path.join(SAVE_DIR, "training_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"Metadata → {SAVE_DIR}/training_metadata.json")

print("""
\n--- Cara Load Ulang ---
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model = AutoModelForTokenClassification.from_pretrained(SAVE_DIR)
tok   = AutoTokenizer.from_pretrained(SAVE_DIR)
pipe  = pipeline("token-classification", model=model, tokenizer=tok,
                 aggregation_strategy="simple")
""")
