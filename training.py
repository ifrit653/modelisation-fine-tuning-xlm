import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, f1_score
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Load data
with open("absa_dataset_combined.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

# Filter entries with required keys
entries = [e for e in entries if all(k in e for k in ["text", "aspect", "sentiment"])]

# Label mapping
label_map = {"positive": 0, "neutral": 1, "negative": 2}
id2label = {v: k for k, v in label_map.items()}

print(f"Total entries: {len(entries)}")
print("Label distribution:")
label_counts = {}
for entry in entries:
    label = entry["sentiment"]
    label_counts[label] = label_counts.get(label, 0) + 1
print(label_counts)

# Initialize tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

class ABSADataset(Dataset):
    def __init__(self, entries, tokenizer, label_map, max_len=128):
        self.entries = entries
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        text = item["text"]
        aspect = item["aspect"]
        label = self.label_map[item["sentiment"]]

        # Combine text and aspect with a separator
        encoded = self.tokenizer(
            f"{text} [SEP] {aspect}",  # Using [SEP] instead of [ASP] which might not exist
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Split data
from sklearn.model_selection import train_test_split
train_entries, val_entries = train_test_split(
    entries, 
    test_size=0.1, 
    stratify=[e["sentiment"] for e in entries], 
    random_state=42
)

train_dataset = ABSADataset(train_entries, tokenizer, label_map)
val_dataset = ABSADataset(val_entries, tokenizer, label_map)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Load model
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=3,
    id2label=id2label,
    label2id=label_map
)

# Improved compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    # Calculate metrics
    report = classification_report(labels, preds, target_names=list(label_map.keys()), output_dict=True, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    
    # Calculate macro and weighted F1 scores
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    conf_mat = confusion_matrix(labels, preds)

    print("\nConfusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=list(label_map.keys()), zero_division=0))
    print(f"MCC: {mcc:.3f}, Kappa: {kappa:.3f}")
    print(f"Macro F1: {macro_f1:.3f}, Weighted F1: {weighted_f1:.3f}")

    return {
        "accuracy": report["accuracy"],
        "f1_macro": macro_f1,  # This is what we'll use for best model selection
        "f1_weighted": weighted_f1,
        "f1_positive": report["positive"]["f1-score"],
        "f1_neutral": report["neutral"]["f1-score"],
        "f1_negative": report["negative"]["f1-score"],
        "mcc": mcc,
        "kappa": kappa
    }

# Training arguments with fixes
training_args = TrainingArguments(
    output_dir="./absa-checkpoints",
    eval_strategy="epoch",  # Fixed the deprecated parameter name
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,  # Reduced batch size to help with class imbalance
    per_device_eval_batch_size=8,
    num_train_epochs=5,  # Increased epochs
    learning_rate=2e-5,  # Added explicit learning rate
    weight_decay=0.01,
    warmup_ratio=0.1,  # Added warmup
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",  # Fixed to use the correct metric name
    greater_is_better=True,
    report_to="none",
    logging_steps=50,
    save_total_limit=2,  # Only keep 2 best checkpoints
    seed=42
)

# Calculate class weights to handle imbalance
from collections import Counter
label_counts = Counter([e["sentiment"] for e in train_entries])
total_samples = len(train_entries)
class_weights = {}
for label, count in label_counts.items():
    class_weights[label_map[label]] = total_samples / (len(label_map) * count)

print("Class weights:", class_weights)

# Custom Trainer class to handle class weights
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Apply class weights
        weight_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))]).to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Initialize trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,  # Use processing_class instead of tokenizer
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# Final evaluation
print("Running final evaluation...")
eval_results = trainer.evaluate()
print("\nFinal Evaluation Results:", eval_results)

# Save the model
model.save_pretrained("./final-absa-model")
tokenizer.save_pretrained("./final-absa-model")
print("Model saved to ./final-absa-model")