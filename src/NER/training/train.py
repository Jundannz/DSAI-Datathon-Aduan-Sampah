model = AutoModelForTokenClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=CONFIG["label2id"],
    hidden_dropout_prob=CONFIG["dropout"],
    attention_probs_dropout_prob=CONFIG["dropout"] * 0.5,
)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model             : {CONFIG['model_name']}")
print(f"Total params      : {total_params:,}")
print(f"Trainable params  : {trainable_params:,}")
print(f"Model size (est.) : {total_params * 4 / 1e6:.0f} MB (float32)")
print()
print("Pesan 'missing keys' di atas = NORMAL (classification head random init)")

class WeightedLossTrainer(Trainer):
    """
    Trainer dengan CrossEntropyLoss berbobot.
    Gradient dari B-LOC/I-LOC diperkuat ~4-5× vs O.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits  # (batch, seq_len, num_labels)

        weights = CLASS_WEIGHTS.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

        loss = loss_fn(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


print("WeightedLossTrainer terdefinisi.")
print(f"  Loss: CrossEntropyLoss(weight=[{CLASS_WEIGHTS[0]:.3f}, {CLASS_WEIGHTS[1]:.3f}, {CLASS_WEIGHTS[2]:.3f}])")
print(f"  Efek: token B-LOC memberikan {CLASS_WEIGHTS[1]:.1f}× lebih besar gradient dari O")

def compute_metrics(eval_pred):
    """Evaluasi span-level (entitas utuh, bukan token individual)."""
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        tl, tp = [], []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            tl.append(ID2LABEL[label_id])
            tp.append(ID2LABEL[pred_id])
        true_labels.append(tl)
        true_preds.append(tp)

    p  = seqeval_precision(true_labels, true_preds, average="weighted", zero_division=0)
    r  = seqeval_recall(   true_labels, true_preds, average="weighted", zero_division=0)
    f1 = seqeval_f1(       true_labels, true_preds, average="weighted", zero_division=0)

    try:
        full_report = seqeval_report(true_labels, true_preds, output_dict=True, zero_division=0)
        loc_f1 = full_report.get("LOC", {}).get("f1-score",  0.0)
        loc_p  = full_report.get("LOC", {}).get("precision", 0.0)
        loc_r  = full_report.get("LOC", {}).get("recall",    0.0)
    except Exception:
        loc_f1 = loc_p = loc_r = 0.0

    return {
        "precision"    : round(p,      4),
        "recall"       : round(r,      4),
        "f1"           : round(f1,     4),
        "f1_LOC"       : round(loc_f1, 4),
        "precision_LOC": round(loc_p,  4),
        "recall_LOC"   : round(loc_r,  4),
    }


print("compute_metrics terdefinisi (seqeval span-level, early stop via f1_LOC)")

use_fp16 = torch.cuda.is_available()

total_steps = (len(train_tok) // CONFIG["batch_size"]) * CONFIG["num_epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],

    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"] * 2,

    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    max_grad_norm=CONFIG["max_grad_norm"],
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_LOC",
    greater_is_better=True,

    fp16=use_fp16,
    dataloader_num_workers=2,
    group_by_length=True,

    seed=CONFIG["seed"],
    data_seed=CONFIG["seed"],
    report_to="none",
    save_total_limit=3,
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stop_patience"])],
)

total_steps  = len(train_tok) // CONFIG["batch_size"] * CONFIG["num_epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
print(f"WeightedLossTrainer siap.")
print(f"  Total steps  : {total_steps}")
print(f"  Warmup steps : {warmup_steps}")
print(f"  FP16         : {use_fp16}")
print(f"  Early stop   : patience={CONFIG['early_stop_patience']} epoch")

print("Mulai training...\n")
train_result = trainer.train()

print("\n" + "=" * 60)
print("TRAINING SELESAI")
print("=" * 60)
print(f"  Train runtime  : {train_result.metrics.get('train_runtime', 0):.0f} s")
print(f"  Train loss     : {train_result.metrics.get('train_loss', 0):.4f}")
print(f"  Samples/second : {train_result.metrics.get('train_samples_per_second', 0):.1f}")

import matplotlib.pyplot as plt

log_history = trainer.state.log_history

epochs_train, train_losses = [], []
epochs_eval, eval_losses   = [], []
epochs_f1, eval_f1s        = [], []
epochs_loc, eval_loc_f1s   = [], []

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        epochs_train.append(entry["epoch"])
        train_losses.append(entry["loss"])
    if "eval_loss" in entry:
        epochs_eval.append(entry["epoch"])
        eval_losses.append(entry["eval_loss"])
        eval_f1s.append(entry.get("eval_f1", 0))
        eval_loc_f1s.append(entry.get("eval_f1_LOC", 0))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(epochs_train, train_losses, label="Train Loss", color="#3498db", marker="o")
axes[0].plot(epochs_eval,  eval_losses,  label="Val Loss",   color="#e74c3c", marker="s")
axes[0].set_title("Loss Curve", fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_eval, eval_f1s, label="Val F1 (weighted)", color="#2ecc71", marker="o")
if eval_f1s:
    axes[1].axhline(max(eval_f1s), color="gray", linestyle="--", alpha=0.6,
                    label=f"Best={max(eval_f1s):.4f}")
axes[1].set_title("Val F1 Weighted", fontweight="bold")
axes[1].set_ylim(0, 1.05); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(epochs_eval, eval_loc_f1s, label="Val F1 LOC", color="#9b59b6", marker="o")
if eval_loc_f1s:
    axes[2].axhline(max(eval_loc_f1s), color="gray", linestyle="--", alpha=0.6,
                    label=f"Best={max(eval_loc_f1s):.4f}")
axes[2].set_title("Val F1 Entity LOC", fontweight="bold")
axes[2].set_ylim(0, 1.05); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle(f"NER Training Curves — {CONFIG['model_name']}", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout(); plt.show()

print("\n" + "=" * 60)
best_f1_loc = max(eval_loc_f1s) if eval_loc_f1s else 0
best_epoch  = (eval_loc_f1s.index(best_f1_loc) + 1) if eval_loc_f1s else 0
stopped_at  = len(eval_losses)
print(f"  Best Val F1 LOC  : {best_f1_loc:.4f} (epoch {best_epoch})")
print(f"  Stopped at epoch : {stopped_at}/{CONFIG['num_epochs']}")
if best_f1_loc >= 0.80: print("  STATUS: SANGAT BAIK — siap deploy")
elif best_f1_loc >= 0.65: print("  STATUS: BAIK — bisa ditingkatkan dengan data augmentasi")
else: print("  STATUS: PERLU REVIEW")
print("=" * 60)
