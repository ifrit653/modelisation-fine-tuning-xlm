import torch
from torch.utils.data import Dataset
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import json
import random

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ==============================
# Dataset Class
# ==============================
class AspectDataset(Dataset):
    def __init__(self, texts, aspects, tokenizer, aspect2id, max_len=128):
        self.texts = texts
        self.aspects = aspects
        self.tokenizer = tokenizer
        self.aspect2id = aspect2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.aspect2id[self.aspects[idx]], dtype=torch.long)
        }


# ==============================
# Trainer Class
# ==============================
class AspectTrainer:
    def __init__(self, model_name="xlm-roberta-base"):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.aspect2id, self.id2aspect, self.class_weights = {}, {}, {}

    def load_data(self, file="absa_dataset_consolidated.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [d["text"] for d in data]
        aspects = [d["aspect"] for d in data]
        return texts, aspects

    def prepare_mappings(self, aspects):
        counts = Counter(aspects)
        self.aspect2id = {a: i for i, a in enumerate(counts.keys())}
        self.id2aspect = {i: a for a, i in self.aspect2id.items()}
        total = len(aspects)
        self.class_weights = {
            self.aspect2id[a]: np.sqrt(total / (len(counts) * c))
            for a, c in counts.items()
        }
        return counts

    def split(self, texts, aspects):
        # First split (train vs temp) - stratified
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, aspects, test_size=0.25, stratify=aspects, random_state=42
        )

        # Second split (val vs test) - NO stratify to avoid "only 1 sample" error
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.4, random_state=42
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def compute_metrics(self, pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0)
        }

    def train(self, train_ds, val_ds, output="./aspect-model"):
        model = XLMRobertaForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.aspect2id),
            id2label=self.id2aspect, label2id=self.aspect2id
        )

        args = TrainingArguments(
            output_dir=output,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_dir="./logs",
            save_total_limit=2,
            seed=42
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        model.save_pretrained(output)
        self.tokenizer.save_pretrained(output)
        return trainer


# ==============================
# Helper to filter rare aspects
# ==============================
def filter_rare_aspects(texts, aspects, min_samples=3):
    counts = Counter(aspects)
    keep_idx = [i for i, a in enumerate(aspects) if counts[a] >= min_samples]
    return [texts[i] for i in keep_idx], [aspects[i] for i in keep_idx]


# ==============================
# Main Script
# ==============================
def main():
    trainer = AspectTrainer()
    texts, aspects = trainer.load_data()

    # Filter rare classes
    texts, aspects = filter_rare_aspects(texts, aspects, min_samples=3)

    counts = trainer.prepare_mappings(aspects)
    print(f"Number of aspects after filtering: {len(counts)}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.split(texts, aspects)

    train_ds = AspectDataset(X_train, y_train, trainer.tokenizer, trainer.aspect2id)
    val_ds = AspectDataset(X_val, y_val, trainer.tokenizer, trainer.aspect2id)
    test_ds = AspectDataset(X_test, y_test, trainer.tokenizer, trainer.aspect2id)

    model_trainer = trainer.train(train_ds, val_ds)

    # Final evaluation
    results = model_trainer.evaluate(test_ds)
    print("\nFinal Test Results:", results)


if __name__ == "__main__":
    main()
