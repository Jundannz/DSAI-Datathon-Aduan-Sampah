# ── Metrik evaluasi menggunakan library evaluate ──────────────────────────────
_acc_metric  = evaluate.load("accuracy")
_f1_metric   = evaluate.load("f1")
_prec_metric = evaluate.load("precision")
_rec_metric  = evaluate.load("recall")

def compute_metrics(eval_pred) -> dict:
    """
    Dipanggil di akhir setiap epoch eval oleh Trainer.
    Mengembalikan F1-macro sebagai metrik utama plus Accuracy,
    Precision, dan Recall untuk monitoring komprehensif.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro"       : round(_f1_metric.compute(
                               predictions=preds, references=labels, average="macro")["f1"], 4),
        "accuracy"       : round(_acc_metric.compute(
                               predictions=preds, references=labels)["accuracy"], 4),
        "precision_macro": round(_prec_metric.compute(
                               predictions=preds, references=labels,
                               average="macro", zero_division=0)["precision"], 4),
        "recall_macro"   : round(_rec_metric.compute(
                               predictions=preds, references=labels,
                               average="macro", zero_division=0)["recall"], 4),
    }


# ── model_init — dipanggil ulang setiap trial Optuna ─────────────────────────
def model_init(trial=None) -> AutoModelForSequenceClassification:
    """
    Membuat instance model yang bersih di setiap pemanggilan.
    Diperlukan oleh hyperparameter_search agar setiap trial
    dimulai dari bobot awal yang sama (tidak mewarisi state trial sebelumnya).

    Dropout dinaikkan dari default IndoBERT (0.1) ke 0.3 untuk regularisasi
    tambahan mengingat ukuran dataset yang relatif kecil.
    """
    mdl = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels              = CONFIG["num_labels"],
        id2label                = ID2LABEL,
        label2id                = LABEL2ID,
        ignore_mismatched_sizes = True,
    )
    mdl.config.hidden_dropout_prob          = CONFIG["dropout"]
    mdl.config.attention_probs_dropout_prob = CONFIG["dropout"] / 2
    for layer in mdl.bert.encoder.layer:
        layer.attention.self.dropout.p   = mdl.config.attention_probs_dropout_prob
        layer.attention.output.dropout.p = mdl.config.hidden_dropout_prob
        layer.output.dropout.p           = mdl.config.hidden_dropout_prob
    return mdl


# ── CustomTrainer dengan Weighted CrossEntropyLoss ────────────────────────────
class CustomTrainer(Trainer):
    """
    Subclass Trainer yang meng-override compute_loss untuk menyuntikkan
    class weights ke CrossEntropyLoss.

    Kelas minoritas mendapat gradient yang lebih besar, mencegah model
    bias ke kelas mayoritas — penting untuk dataset yang tidak seimbang.

    CLASS_WEIGHTS_TENSOR harus didefinisikan sebelum trainer dibuat (Cell 7).
    """

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        # Pindahkan weight tensor ke device yang sama dengan logits
        # (penting untuk multi-GPU atau saat device berubah antar trial)
        loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS_TENSOR.to(logits.device))
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


print("Setup selesai:")
print(f"  Metrik utama     : Macro F1 (eval_f1_macro)")
print(f"  Metrik tambahan  : Accuracy, Precision (macro), Recall (macro)")
print(f"  Loss function    : CrossEntropyLoss dengan class weights")
print(f"  Dropout override : hidden={CONFIG['dropout']}, attn={CONFIG['dropout']/2}")
print(f"  model_init       : Siap untuk Optuna hyperparameter_search")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs(CONFIG["output_dir"] + "/hp_search", exist_ok=True)

# ── Hyperparameter Search Space ───────────────────────────────────────────────
# Tiga parameter yang paling berdampak untuk fine-tuning BERT:
#   - learning_rate   : faktor terpenting, log-scale untuk eksplorasi rentang lebar
#   - weight_decay    : regularisasi L2 untuk mencegah overfit
#   - num_train_epochs: durasi training per trial (dibatasi 3-5 agar efisien)
#
# Batch size sengaja tidak dimasukkan ke search space agar trial tidak terlalu
# bervariasi (pada GPU Kaggle, batch size 16 sudah optimal untuk IndoBERT-base).

def hp_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate"   : trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "weight_decay"    : trial.suggest_float("weight_decay", 0.01, 0.30),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
    }

# ── TrainingArguments untuk fase search ──────────────────────────────────────
# save_strategy="no" agar tidak ada checkpoint per trial — hemat disk.
# Nilai default di bawah akan di-override oleh Optuna per trial.
search_args = TrainingArguments(
    output_dir                  = CONFIG["output_dir"] + "/hp_search",
    per_device_train_batch_size = CONFIG["batch_size"],
    per_device_eval_batch_size  = CONFIG["batch_size"],
    warmup_ratio                = CONFIG["warmup_ratio"],
    lr_scheduler_type           = "linear",
    eval_strategy               = "epoch",
    save_strategy               = "no",
    logging_steps               = 100,
    report_to                   = "none",
    seed                        = CONFIG["seed"],
    data_seed                   = CONFIG["seed"],
    fp16                        = torch.cuda.is_available(),
    dataloader_num_workers      = 2,
    # Nilai default — akan di-override oleh Optuna
    learning_rate               = 2e-5,
    weight_decay                = 0.1,
    num_train_epochs            = CONFIG["search_epochs"],
)

# CustomTrainer dengan model_init untuk search
search_trainer = CustomTrainer(
    args            = search_args,
    model_init      = model_init,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    processing_class= tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

print(f"Memulai Optuna hyperparameter search ({CONFIG['n_trials']} trials)...")
print(f"Setiap trial melatih model baru dengan hyperparameter berbeda.\n")

best_run = search_trainer.hyperparameter_search(
    direction        = "maximize",
    backend          = "optuna",
    n_trials         = CONFIG["n_trials"],
    hp_space         = hp_space,
    compute_objective= lambda metrics: metrics["eval_f1_macro"],
)

print("\nHyperparameter Search Selesai!")
print(f"  Best objective (F1-macro) : {best_run.objective:.4f}")
print(f"  Best hyperparameters:")
for k, v in best_run.hyperparameters.items():
    print(f"    {k:<30} : {v}")

best_hp = best_run.hyperparameters

os.makedirs(CONFIG["output_dir"] + "/final", exist_ok=True)

# ── TrainingArguments final menggunakan best hyperparams dari Optuna ──────────
# load_best_model_at_end=True memastikan checkpoint terbaik (berdasarkan
# eval_f1_macro) yang dimuat di akhir, bukan checkpoint epoch terakhir.
final_args = TrainingArguments(
    output_dir                  = CONFIG["output_dir"] + "/final",
    num_train_epochs            = CONFIG["num_epochs"],
    per_device_train_batch_size = CONFIG["batch_size"],
    per_device_eval_batch_size  = CONFIG["batch_size"],
    learning_rate               = best_hp.get("learning_rate", 2e-5),
    weight_decay                = best_hp.get("weight_decay", 0.1),
    warmup_ratio                = CONFIG["warmup_ratio"],
    lr_scheduler_type           = "linear",
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_f1_macro",
    greater_is_better           = True,
    save_total_limit            = 2,
    logging_steps               = 20,
    report_to                   = "none",
    seed                        = CONFIG["seed"],
    data_seed                   = CONFIG["seed"],
    fp16                        = torch.cuda.is_available(),
    dataloader_num_workers      = 2,
)

# Fresh model — mulai dari bobot pretrained, bukan dari state trial sebelumnya
final_model = model_init()

final_trainer = CustomTrainer(
    model           = final_model,
    args            = final_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    processing_class= tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Konfigurasi Final Training:")
print(f"  Max Epochs       : {CONFIG['num_epochs']} (early stop patience=2)")
print(f"  Learning Rate    : {final_args.learning_rate:.2e}  (dari Optuna)")
print(f"  Weight Decay     : {final_args.weight_decay:.4f}  (dari Optuna)")
print(f"  Batch Size       : {final_args.per_device_train_batch_size}")
print(f"  Warmup Ratio     : {final_args.warmup_ratio}")
print(f"  Best Model By    : eval_f1_macro")
print(f"  Mixed Precision  : {torch.cuda.is_available()}\n")

print("Memulai final training...")
train_result = final_trainer.train()

print("\nFinal Training Selesai!")
print(f"  Runtime          : {train_result.metrics.get('train_runtime', 0):.1f}s")
print(f"  Train Loss       : {train_result.metrics.get('train_loss', 0):.4f}")
print(f"  Samples/second   : {train_result.metrics.get('train_samples_per_second', 0):.1f}")

# ── Ambil log history dari trainer ───────────────────────────────────────────
history     = final_trainer.state.log_history
train_logs  = [h for h in history if "loss" in h and "eval_loss" not in h]
eval_logs   = [h for h in history if "eval_loss" in h]

train_losses = [h["loss"]              for h in train_logs]
eval_losses  = [h["eval_loss"]         for h in eval_logs]
eval_f1s     = [h.get("eval_f1_macro", 0)    for h in eval_logs]
eval_accs    = [h.get("eval_accuracy", 0)    for h in eval_logs]
eval_precs   = [h.get("eval_precision_macro", 0) for h in eval_logs]
eval_recs    = [h.get("eval_recall_macro", 0)    for h in eval_logs]
epochs_list  = [h["epoch"]             for h in eval_logs]

# ── Plot training curves ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
train_steps = [h["step"] for h in train_logs]
axes[0].plot(train_steps, train_losses, label="Train Loss",
             color="#3498db", alpha=0.7, linewidth=1.5)
if eval_logs:
    step_per_epoch = max(1, len(train_logs) // max(1, len(eval_logs)))
    eval_steps_plt = [i * step_per_epoch for i in range(1, len(eval_logs) + 1)]
    axes[0].plot(eval_steps_plt, eval_losses, label="Val Loss",
                 color="#e74c3c", linewidth=2, marker="o", markersize=6)
axes[0].set_title("Loss Curve", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Training Steps", fontsize=11)
axes[0].set_ylabel("Loss", fontsize=11)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Metrics curve
axes[1].plot(epochs_list, eval_f1s,   label="Val F1 (macro)",       color="#2ecc71",  linewidth=2, marker="o")
axes[1].plot(epochs_list, eval_accs,  label="Val Accuracy",          color="#9b59b6",  linewidth=2, marker="s")
axes[1].plot(epochs_list, eval_precs, label="Val Precision (macro)", color="#f39c12",  linewidth=2, marker="^", linestyle="--")
axes[1].plot(epochs_list, eval_recs,  label="Val Recall (macro)",    color="#1abc9c",  linewidth=2, marker="D", linestyle="--")
axes[1].set_title("Validation Metrics per Epoch", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("Score", fontsize=11)
axes[1].set_ylim(0, 1.05)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.suptitle("Training Curves — Final Model", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# ── Training Status Monitor ───────────────────────────────────────────────────
# Mendeteksi pola OVERFIT / UNDERFIT / NORMAL dari riwayat training.
print("=" * 60)
print("  TRAINING STATUS MONITOR")
print("=" * 60)

if eval_logs and train_logs:
    final_train_loss = train_losses[-1]
    final_val_loss   = eval_losses[-1]
    final_val_f1     = eval_f1s[-1]
    best_val_f1      = max(eval_f1s)
    rel_gap          = (final_val_loss - final_train_loss) / (final_train_loss + 1e-9)

    print(f"  Final Train Loss    : {final_train_loss:.4f}")
    print(f"  Final Val Loss      : {final_val_loss:.4f}")
    print(f"  Relative Gap        : {rel_gap:.2f}")
    print(f"  Best Val F1-macro   : {best_val_f1:.4f}")
    print(f"  Final Val F1-macro  : {final_val_f1:.4f}")
    print()

    # Deteksi status
    if final_train_loss > 0.6 and final_val_f1 < 0.5:
        STATUS = "UNDERFIT"
        DESC   = ("Model belum cukup belajar. "
                  "Coba: tambah epoch, kurangi dropout, atau perbesar dataset.")
    elif rel_gap > 0.5 and final_val_f1 < best_val_f1 - 0.05:
        STATUS = "OVERFIT"
        DESC   = ("Val loss jauh di atas train loss dan F1 turun di akhir. "
                  "Early stopping sudah bekerja. "
                  "Pertimbangkan: naikkan dropout, atau tambah data asli.")
    elif final_val_f1 >= 0.70:
        STATUS = "NORMAL"
        DESC   = f"Training berjalan baik. Best F1-macro = {best_val_f1:.4f}."
    elif final_val_f1 >= 0.50:
        STATUS = "NORMAL (belum optimal)"
        DESC   = "Training stabil namun F1 masih dapat ditingkatkan melalui iterasi."
    else:
        STATUS = "PERLU INVESTIGASI"
        DESC   = ("F1 rendah. Kemungkinan: masalah pada label, distribusi kelas, "
                  "atau teks terlalu pendek untuk konteks model.")

    print(f"  STATUS   : {STATUS}")
    print(f"  ANALISIS : {DESC}")

    # Tips berdasarkan trend
    if len(eval_losses) >= 3 and eval_losses[-1] > eval_losses[-2]:
        print("  Catatan  : Val loss naik di epoch terakhir — early stopping sudah bekerja dengan benar.")
    if len(eval_f1s) >= 2 and eval_f1s[-1] > eval_f1s[-2]:
        print("  Catatan  : Val F1 masih naik di epoch terakhir — pertimbangkan menambah 1-2 epoch.")
else:
    print("  Log history tidak tersedia — jalankan ulang Cell 11.")

print("=" * 60)
