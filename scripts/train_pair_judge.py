"""
Train a binary "does this PMI unit cover this TZ requirement?" cross-encoder.

Input:
    data/c_pairs_variant_a.csv  — 14,794 pre-labelled (tz_req_text,
    pmi_unit_text, label) pairs produced by the earlier generate_pairs
    pipeline. Distribution:
        pseudo_pos   (label=1)   1,237   — fuzzy-matched positives
        rand_neg     (label=0)   6,185   — random pairs
        xpkg_neg     (label=0)   3,711   — cross-package pairs
        hard_neg     (label=0)   3,661   — high retrieval / wrong topic

Architecture:
    Fine-tune a pair cross-encoder — one transformer that sees
    (query, passage) jointly and outputs a relevance logit. Base:
    bert-base-multilingual-cased (reuses the same checkpoint that
    req_classifier was built on, keeps dependencies minimal). At inference
    the new judge (CrossEncoderCoverageJudge) converts P(match) into a
    CoverageStatus-compatible label.

Splits:
    Stratified by pkg_id (test pairs come from packages not seen during
    training) so we never leak fuzzy-matched positives from one package
    into the val/test set. 80/10/10 by package.

Usage:
    # Eval-only baseline (no training)
    python scripts/train_pair_judge.py --eval-only \
        --base C:/Users/Marilka/PycharmProjects/requirement-model \
        --csv  ./data/c_pairs_variant_a.csv

    # Fine-tune
    python scripts/train_pair_judge.py \
        --base C:/Users/Marilka/PycharmProjects/requirement-model \
        --csv  ./data/c_pairs_variant_a.csv \
        --out  ./model/pair_judge \
        --batch 16 --grad-accum 2 --epochs 3 --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
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


class WeightedTrainer(Trainer):
    """Trainer that applies a class-weighted CrossEntropyLoss.

    Rationale: the pair dataset is ~8% positives, so the default CE loss is
    dominated by easy negatives and the model underfits the MATCH class
    (see v1 results: rand_neg accuracy 99.8% but pseudo_pos accuracy only
    58.5%). Up-weighting the positive class by its inverse frequency tells
    the optimiser that each positive mistake is ~11× more costly than a
    negative mistake, which in practice bumps recall on MATCH at a modest
    precision cost.

    `class_weights` is a Python list of length 2 (NO_MATCH, MATCH). If not
    provided or not positive, falls back to vanilla CE.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self._class_weights is not None:
            weights = torch.tensor(
                self._class_weights, device=logits.device, dtype=logits.dtype
            )
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256) -> None:
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        enc = self.tok(
            str(row["tz_req_text"]),
            str(row["pmi_unit_text"]),
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        enc["labels"] = int(row["label"])
        return enc


def _stratified_split_by_package(
    df: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10 by pkg_id so each package is in exactly one split."""
    rng = random.Random(seed)
    packages = sorted(df["pkg_id"].unique().tolist())
    rng.shuffle(packages)
    n = len(packages)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    train_pkgs = set(packages[:n_train])
    val_pkgs = set(packages[n_train : n_train + n_val])
    test_pkgs = set(packages[n_train + n_val :])
    train = df[df["pkg_id"].isin(train_pkgs)]
    val = df[df["pkg_id"].isin(val_pkgs)]
    test = df[df["pkg_id"].isin(test_pkgs)]
    return train, val, test


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


def full_report(
    model, tokenizer, df: pd.DataFrame, device: str, batch: int, max_len: int,
) -> Dict:
    model.eval()
    model.to(device)

    reqs = df["tz_req_text"].astype(str).tolist()
    units = df["pmi_unit_text"].astype(str).tolist()
    labels = np.array(df["label"].astype(int).tolist())
    probs_pos: List[float] = []
    preds: List[int] = []

    with torch.no_grad():
        for i in range(0, len(reqs), batch):
            enc = tokenizer(
                reqs[i : i + batch],
                units[i : i + batch],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_pos.extend(probs[:, 1].tolist())
            preds.extend(probs.argmax(axis=-1).tolist())

    preds_arr = np.array(preds)
    report = classification_report(
        labels, preds_arr, target_names=["no_match", "match"],
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(labels, preds_arr).tolist()

    sweeps: Dict[str, Dict[str, float]] = {}
    probs_arr = np.array(probs_pos)
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        pr = (probs_arr >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, pr, average="binary", zero_division=0, pos_label=1,
        )
        sweeps[f"thr={thr:.2f}"] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
        }

    # Break down by pair_type if present — tells us whether the model
    # handles each negative bucket differently (important: hard_neg is the
    # interesting one; rand_neg is trivial; pseudo_pos is gold by construction).
    by_bucket: Dict[str, Dict[str, float]] = {}
    if "pair_type" in df.columns:
        for bucket in sorted(df["pair_type"].unique()):
            mask = (df["pair_type"] == bucket).values
            if not mask.any():
                continue
            bl = labels[mask]
            bp = preds_arr[mask]
            if len(set(bl)) < 2:
                # Single-class bucket — only precision or only recall is meaningful.
                correct = int((bl == bp).sum())
                by_bucket[bucket] = {
                    "n": int(mask.sum()),
                    "accuracy": round(correct / max(1, mask.sum()), 4),
                    "note": "single-class bucket",
                }
                continue
            p_b, r_b, f1_b, _ = precision_recall_fscore_support(
                bl, bp, average="binary", zero_division=0, pos_label=1,
            )
            by_bucket[bucket] = {
                "n": int(mask.sum()),
                "precision": round(float(p_b), 4),
                "recall": round(float(r_b), 4),
                "f1": round(float(f1_b), 4),
            }

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "threshold_sweep": sweeps,
        "by_pair_type": by_bucket,
        "n": int(len(labels)),
        "pos_fraction_gold": float(labels.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="Base checkpoint to fine-tune from. Accepts local "
                         "path or HF hub name (e.g. 'xlm-roberta-base').")
    ap.add_argument("--csv", default="./data/c_pairs_variant_a.csv")
    ap.add_argument("--out", default="./model/pair_judge")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="Weight for the MATCH class in the CE loss. Omit or "
                         "set 'auto' below. When 'auto' is used the weight "
                         "is neg_count/pos_count, which for our v1 dataset is "
                         "~11. A hand-tuned value is often slightly lower.")
    ap.add_argument("--auto-pos-weight", action="store_true",
                    help="Shortcut to set pos_weight = |neg|/|pos| from train.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Defensive: drop rows with empty texts (shouldn't happen but wrecks tokenizer)
    df = df[df["tz_req_text"].notna() & df["pmi_unit_text"].notna()]
    df = df[df["tz_req_text"].str.strip().ne("") & df["pmi_unit_text"].str.strip().ne("")]

    train_df, val_df, test_df = _stratified_split_by_package(df, args.seed)
    print(f"[info] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"[info] train pkgs={train_df['pkg_id'].nunique()}  "
          f"val={val_df['pkg_id'].nunique()}  test={test_df['pkg_id'].nunique()}")
    print(f"[info] train positives={train_df['label'].sum()}/{len(train_df)} "
          f"({train_df['label'].mean():.1%})")

    print(f"[info] loading tokenizer + model from {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)
    model.config.id2label = {0: "NO_MATCH", 1: "MATCH"}
    model.config.label2id = {"NO_MATCH": 0, "MATCH": 1}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    print("\n[baseline] evaluating base model on val + test BEFORE training")
    baseline_val = full_report(model, tok, val_df, device, args.batch, args.max_len)
    baseline_test = full_report(model, tok, test_df, device, args.batch, args.max_len)
    print(f"[baseline] val  F1={baseline_val['classification_report']['match']['f1-score']:.4f}  "
          f"P={baseline_val['classification_report']['match']['precision']:.4f}  "
          f"R={baseline_val['classification_report']['match']['recall']:.4f}")
    print(f"[baseline] test F1={baseline_test['classification_report']['match']['f1-score']:.4f}  "
          f"P={baseline_test['classification_report']['match']['precision']:.4f}  "
          f"R={baseline_test['classification_report']['match']['recall']:.4f}")

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

    ds_train = PairDataset(train_df, tok, max_len=args.max_len)
    ds_val = PairDataset(val_df, tok, max_len=args.max_len)

    # Compute class weights for the MATCH class. Rule: weight_pos = |neg| / |pos|
    # for "auto" mode. The NO_MATCH class weight stays at 1.0.
    class_weights = None
    if args.auto_pos_weight or args.pos_weight is not None:
        if args.auto_pos_weight:
            n_pos = int(train_df["label"].sum())
            n_neg = len(train_df) - n_pos
            pos_w = n_neg / max(1, n_pos)
        else:
            pos_w = float(args.pos_weight)
        class_weights = [1.0, pos_w]
        print(f"[info] Using weighted CE loss: NO_MATCH=1.0 MATCH={pos_w:.2f}")

    training_args = TrainingArguments(
        output_dir=str(out_dir / "hf_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(args.batch, 16),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        report_to=["none"],
        fp16=(device == "cuda"),
    )

    # Use WeightedTrainer when class_weights is set; otherwise plain Trainer.
    # transformers 5.0 renamed tokenizer= → processing_class=.
    trainer_cls = WeightedTrainer if class_weights is not None else Trainer
    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=ds_train, eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    if class_weights is not None:
        trainer_kwargs["class_weights"] = class_weights
    try:
        trainer = trainer_cls(processing_class=tok, **trainer_kwargs)
    except TypeError:
        trainer = trainer_cls(tokenizer=tok, **trainer_kwargs)

    print("\n[train] starting fine-tuning")
    trainer.train()
    print("[train] done")

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[train] best model saved to {out_dir}")

    print("\n[final] evaluating fine-tuned model on test")
    final_test = full_report(model, tok, test_df, device, args.batch, args.max_len)
    print(f"[final] test F1={final_test['classification_report']['match']['f1-score']:.4f}  "
          f"P={final_test['classification_report']['match']['precision']:.4f}  "
          f"R={final_test['classification_report']['match']['recall']:.4f}")
    print("[final] threshold sweep:")
    for thr, m in final_test["threshold_sweep"].items():
        print(f"    {thr}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("[final] by pair_type:")
    for bucket, m in final_test["by_pair_type"].items():
        extras = " ".join(f"{k}={v}" for k, v in m.items() if k not in ("n",))
        print(f"    {bucket:12s} (n={m['n']}): {extras}")
    (out_dir / "final_test.json").write_text(
        json.dumps(final_test, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
