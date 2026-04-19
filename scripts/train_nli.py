"""Train a 3-class NLI cross-encoder on labeled pairs.

Input CSV must contain:
- tz_text
- cand_text
- label in {entails, neutral, contradicts}

Usage:
python scripts/train_nli.py --csv ./data/pairs_labeled.csv --out_dir ./models/nli_ce
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

LABEL2ID = {"entails": 0, "neutral": 1, "contradicts": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class PairsDS(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(
            str(r.tz_text),
            str(r.cand_text),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(r.label_id), dtype=torch.long)
        return item

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["label"].isin(LABEL2ID.keys())].copy()
    df["label_id"] = df["label"].map(LABEL2ID)

    strat = df["label_id"] if len(df) > 30 else None
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=strat)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = PairsDS(train_df, tok, max_len=args.max_len)
    val_ds = PairsDS(val_df, tok, max_len=args.max_len)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
    )

    trainer.train()
    preds = trainer.predict(val_ds)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=1)
    print(classification_report(y_true, y_pred, target_names=[ID2LABEL[i] for i in range(3)]))

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Saved model to: {out_dir}")

if __name__ == "__main__":
    main()
