"""
Train (fine-tune) a binary "is requirement" classifier on the JSONL dataset
produced by scripts/build_requirement_dataset.py.

Modes:
  --eval-only    — no training, just measure F1/precision/recall of the base
                   checkpoint on val and test. Use this to get a baseline
                   number from the already-trained model at
                   C:/Users/Marilka/PycharmProjects/requirement-model before
                   spending time fine-tuning.
  (default)      — fine-tune on train.jsonl, early-stop on val F1, evaluate
                   on test.jsonl, save checkpoint to --out.

Label convention: 1 = requirement, 0 = not a requirement.

Usage:
  # Baseline eval of existing checkpoint
  python scripts/train_req_classifier.py --eval-only \
      --base C:/Users/Marilka/PycharmProjects/requirement-model \
      --data ./data/req_dataset

  # Fine-tune locally (GTX 1650 — small batch + accumulation)
  python scripts/train_req_classifier.py \
      --base C:/Users/Marilka/PycharmProjects/requirement-model \
      --data ./data/req_dataset \
      --out ./model/req_classifier \
      --batch 4 --grad-accum 8 --epochs 3 --lr 2e-5

  # Fine-tune on Colab (T4 — larger batch)
  python scripts/train_req_classifier.py \
      --base C:/Users/Marilka/PycharmProjects/requirement-model \
      --data ./data/req_dataset \
      --out ./model/req_classifier \
      --batch 32 --grad-accum 1 --epochs 3 --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ReqDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_len: int = 192) -> None:
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        enc = self.tok(
            r["text"],
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        enc["labels"] = int(r["label"])
        return enc


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0, pos_label=1,
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float((preds == labels).mean()),
    }


def full_report(model, tokenizer, rows: List[dict], device: str, batch: int, max_len: int) -> Dict:
    """Run inference over rows, return classification report + confusion matrix."""
    model.eval()
    model.to(device)

    texts = [r["text"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    preds: List[int] = []
    probs_pos: List[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = tokenizer(
                chunk,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds.extend(probs.argmax(axis=-1).tolist())
            probs_pos.extend(probs[:, 1].tolist())

    preds_arr = np.array(preds)
    report = classification_report(
        labels, preds_arr, target_names=["not_req", "req"],
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(labels, preds_arr).tolist()

    # Threshold sweep — helpful when base model is biased (e.g. existing
    # requirement-model greenlights most sentences at threshold=0.5).
    sweeps: Dict[str, Dict[str, float]] = {}
    probs_arr = np.array(probs_pos)
    for thr in (0.5, 0.7, 0.85, 0.9, 0.95, 0.98):
        pr = (probs_arr >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, pr, average="binary", zero_division=0, pos_label=1,
        )
        sweeps[f"thr={thr:.2f}"] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
        }

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": "rows=gold [not_req, req]; cols=pred [not_req, req]",
        "threshold_sweep": sweeps,
        "n": int(len(labels)),
        "pos_fraction_gold": float(labels.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="Base model checkpoint to fine-tune from (or evaluate).")
    ap.add_argument("--data", default="./data/req_dataset",
                    help="Directory with train.jsonl/val.jsonl/test.jsonl.")
    ap.add_argument("--out", default="./model/req_classifier",
                    help="Where to save the fine-tuned checkpoint.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training, just evaluate --base on val + test.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")

    print(f"[info] train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    print(f"[info] loading tokenizer + model from {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)

    # Ensure human-readable labels are baked into the saved checkpoint so that
    # downstream consumers (ModelRequirementExtractor) don't need to guess.
    model.config.id2label = {0: "NOT_REQUIREMENT", 1: "REQUIREMENT"}
    model.config.label2id = {"NOT_REQUIREMENT": 0, "REQUIREMENT": 1}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    # ------------------------------------------------------------------
    # Baseline (always) — lets us compare before/after fine-tuning even
    # when we are training.
    # ------------------------------------------------------------------

    print("\n[baseline] evaluating base model on val + test BEFORE any training")
    baseline_val = full_report(model, tok, val_rows, device, args.batch, args.max_len)
    baseline_test = full_report(model, tok, test_rows, device, args.batch, args.max_len)
    print(f"[baseline] val  F1={baseline_val['classification_report']['req']['f1-score']:.4f}  "
          f"P={baseline_val['classification_report']['req']['precision']:.4f}  "
          f"R={baseline_val['classification_report']['req']['recall']:.4f}")
    print(f"[baseline] test F1={baseline_test['classification_report']['req']['f1-score']:.4f}  "
          f"P={baseline_test['classification_report']['req']['precision']:.4f}  "
          f"R={baseline_test['classification_report']['req']['recall']:.4f}")
    print(f"[baseline] test threshold sweep:")
    for thr, m in baseline_test["threshold_sweep"].items():
        print(f"    {thr}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_val.json").write_text(
        json.dumps(baseline_val, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "baseline_test.json").write_text(
        json.dumps(baseline_test, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.eval_only:
        print(f"\n[info] --eval-only: skipping training. Reports saved to {out_dir}/")
        return

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    ds_train = ReqDataset(train_rows, tok, max_len=args.max_len)
    ds_val = ReqDataset(val_rows, tok, max_len=args.max_len)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "hf_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(args.batch, 8),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        report_to=["none"],
        fp16=(device == "cuda"),
    )

    # transformers 5.0 renamed `tokenizer=` → `processing_class=`; keep a fallback
    # so the same script works on 4.x as well.
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            processing_class=tok,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            tokenizer=tok,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

    print("\n[train] starting fine-tuning")
    trainer.train()
    print("[train] done")

    # Save best model
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[train] best model saved to {out_dir}")

    # ------------------------------------------------------------------
    # Final evaluation (test set, post-training)
    # ------------------------------------------------------------------

    print("\n[final] evaluating fine-tuned model on test")
    final_test = full_report(model, tok, test_rows, device, args.batch, args.max_len)
    print(f"[final] test F1={final_test['classification_report']['req']['f1-score']:.4f}  "
          f"P={final_test['classification_report']['req']['precision']:.4f}  "
          f"R={final_test['classification_report']['req']['recall']:.4f}")
    print(f"[final] test threshold sweep:")
    for thr, m in final_test["threshold_sweep"].items():
        print(f"    {thr}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    (out_dir / "final_test.json").write_text(
        json.dumps(final_test, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
