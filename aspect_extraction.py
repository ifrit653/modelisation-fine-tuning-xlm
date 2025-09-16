import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForTokenClassification,
    XLMRobertaForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split

# ============================================================================
# APPROACH 1: ASPECT CLASSIFICATION (Multi-class/Multi-label)
# ============================================================================

class AspectClassificationDataset(Dataset):
    """Dataset for aspect classification - predicts which aspects are mentioned in text"""
    
    def __init__(self, entries, tokenizer, aspect_to_id, max_len=128, multi_label=False):
        self.entries = entries
        self.tokenizer = tokenizer
        self.aspect_to_id = aspect_to_id
        self.max_len = max_len
        self.multi_label = multi_label
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        item = self.entries[idx]
        text = item["text"]
        aspects = item["aspect"] if isinstance(item["aspect"], list) else [item["aspect"]]
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        if self.multi_label:
            # Multi-label: can have multiple aspects
            label_vector = torch.zeros(len(self.aspect_to_id))
            for aspect in aspects:
                if aspect in self.aspect_to_id:
                    label_vector[self.aspect_to_id[aspect]] = 1
            labels = label_vector
        else:
            # Single-label: take the first/primary aspect
            primary_aspect = aspects[0] if aspects else "other"
            labels = torch.tensor(self.aspect_to_id.get(primary_aspect, 0), dtype=torch.long)
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": labels
        }

# ============================================================================
# APPROACH 2: ASPECT TERM EXTRACTION (Token Classification with BIO tagging)
# ============================================================================

class AspectTermExtractionDataset(Dataset):
    """Dataset for aspect term extraction using BIO tagging"""
    
    def __init__(self, entries, tokenizer, max_len=128):
        self.entries = self.prepare_bio_data(entries)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_to_id = {"O": 0, "B-ASP": 1, "I-ASP": 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def prepare_bio_data(self, entries):
        """Convert aspect information to BIO tagged sequences"""
        bio_entries = []
        
        for entry in entries:
            text = entry["text"]
            aspects = entry["aspect"] if isinstance(entry["aspect"], list) else [entry["aspect"]]
            
            # Create BIO tags for the text
            tokens = text.split()
            bio_labels = ["O"] * len(tokens)
            
            # Simple aspect matching (you might want to improve this)
            for aspect in aspects:
                aspect_words = aspect.lower().split()
                text_lower = text.lower()
                
                # Find aspect mentions in text
                for i, token in enumerate(tokens):
                    if token.lower() in aspect_words:
                        if bio_labels[i] == "O":  # Don't overwrite existing tags
                            bio_labels[i] = "B-ASP"
                            # Tag subsequent words of multi-word aspects
                            for j in range(i + 1, min(i + len(aspect_words), len(tokens))):
                                if tokens[j].lower() in aspect_words:
                                    bio_labels[j] = "I-ASP"
            
            bio_entries.append({
                "tokens": tokens,
                "labels": bio_labels,
                "original_text": text,
                "aspects": aspects
            })
        
        return bio_entries
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        item = self.entries[idx]
        tokens = item["tokens"]
        labels = item["labels"]
        
        # Tokenize and align labels
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Align labels with tokenized input
        word_ids = encoded.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label_to_id[labels[word_idx]])
            else:
                aligned_labels.append(-100)  # Subword tokens
            previous_word_idx = word_idx
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }

# ============================================================================
# MAIN TRAINING FUNCTIONS
# ============================================================================

def train_aspect_classifier(entries, approach="multi_class"):
    """Train aspect classification model"""
    
    print(f"Training aspect classifier using {approach} approach...")
    
    # Prepare aspects
    all_aspects = []
    for entry in entries:
        aspects = entry["aspect"] if isinstance(entry["aspect"], list) else [entry["aspect"]]
        all_aspects.extend(aspects)
    
    # Create aspect vocabulary
    aspect_counts = Counter(all_aspects)
    print(f"Found {len(aspect_counts)} unique aspects:")
    for aspect, count in aspect_counts.most_common():
        print(f"  {aspect}: {count}")
    
    # Filter aspects with minimum frequency
    min_freq = 5
    common_aspects = [aspect for aspect, count in aspect_counts.items() if count >= min_freq]
    aspect_to_id = {aspect: i for i, aspect in enumerate(common_aspects)}
    id_to_aspect = {i: aspect for aspect, i in aspect_to_id.items()}
    
    print(f"Using {len(common_aspects)} aspects with frequency >= {min_freq}")
    
    # Initialize tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    
    if approach == "multi_label":
        model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(aspect_to_id),
            problem_type="multi_label_classification"
        )
        multi_label = True
    else:
        model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(aspect_to_id),
            id2label=id_to_aspect,
            label2id=aspect_to_id
        )
        multi_label = False
    
    # Prepare datasets
    train_entries, val_entries = train_test_split(entries, test_size=0.1, random_state=42)
    
    train_dataset = AspectClassificationDataset(train_entries, tokenizer, aspect_to_id, multi_label=multi_label)
    val_dataset = AspectClassificationDataset(val_entries, tokenizer, aspect_to_id, multi_label=multi_label)
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        if multi_label:
            predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
            predictions = (predictions > 0.5).astype(int)
            
            # Calculate metrics for multi-label
            f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
            f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
            f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
            
            return {
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }
        else:
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
            f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
            
            print("\nClassification Report:")
            print(classification_report(labels, predictions, target_names=list(aspect_to_id.keys()), zero_division=0))
            
            return {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./aspect-classifier-{approach}",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to="none",
        logging_steps=50,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained(f"./aspect-classifier-{approach}")
    tokenizer.save_pretrained(f"./aspect-classifier-{approach}")
    
    return model, tokenizer, aspect_to_id

def train_aspect_term_extractor(entries):
    """Train aspect term extraction model using token classification"""
    
    print("Training aspect term extraction model...")
    
    # Initialize tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=3,  # O, B-ASP, I-ASP
        id2label={0: "O", 1: "B-ASP", 2: "I-ASP"},
        label2id={"O": 0, "B-ASP": 1, "I-ASP": 2}
    )
    
    # Prepare datasets
    train_entries, val_entries = train_test_split(entries, test_size=0.1, random_state=42)
    
    train_dataset = AspectTermExtractionDataset(train_entries, tokenizer)
    val_dataset = AspectTermExtractionDataset(val_entries, tokenizer)
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens) and flatten
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for pred_i, label_i in zip(prediction, label):
                if label_i != -100:
                    true_predictions.append(pred_i)
                    true_labels.append(label_i)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, true_predictions)
        f1_macro = f1_score(true_labels, true_predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, true_predictions, average='weighted', zero_division=0)
        
        print("\nToken Classification Report:")
        print(classification_report(
            true_labels, 
            true_predictions, 
            target_names=["O", "B-ASP", "I-ASP"], 
            zero_division=0
        ))
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./aspect-term-extractor",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to="none",
        logging_steps=50,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained("./aspect-term-extractor")
    tokenizer.save_pretrained("./aspect-term-extractor")
    
    return model, tokenizer

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def predict_aspects_classification(text, model, tokenizer, aspect_to_id, multi_label=False):
    """Predict aspects using classification model"""
    model.eval()
    
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        
        if multi_label:
            probabilities = torch.sigmoid(logits).squeeze().numpy()
            predicted_aspects = []
            for i, prob in enumerate(probabilities):
                if prob > 0.5:
                    aspect = list(aspect_to_id.keys())[i]
                    predicted_aspects.append((aspect, prob))
            return predicted_aspects
        else:
            predicted_id = torch.argmax(logits, dim=-1).item()
            aspect = list(aspect_to_id.keys())[predicted_id]
            confidence = torch.softmax(logits, dim=-1).squeeze().max().item()
            return [(aspect, confidence)]

def extract_aspect_terms(text, model, tokenizer):
    """Extract aspect terms using token classification model"""
    model.eval()
    
    tokens = text.split()
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**encoded)
        predictions = torch.argmax(outputs.logits, dim=2).squeeze().numpy()
    
    # Align predictions with original tokens
    word_ids = encoded.word_ids(batch_index=0)
    id_to_label = {0: "O", 1: "B-ASP", 2: "I-ASP"}
    
    aspect_terms = []
    current_aspect = []
    
    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            label = id_to_label[predictions[i]]
            
            if label == "B-ASP":
                if current_aspect:
                    aspect_terms.append(" ".join(current_aspect))
                current_aspect = [tokens[word_idx]]
            elif label == "I-ASP" and current_aspect:
                current_aspect.append(tokens[word_idx])
            elif label == "O":
                if current_aspect:
                    aspect_terms.append(" ".join(current_aspect))
                    current_aspect = []
            
            previous_word_idx = word_idx
    
    if current_aspect:
        aspect_terms.append(" ".join(current_aspect))
    
    return aspect_terms

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to train aspect extraction models"""
    
    # Load your data
    print("Loading data...")
    with open("absa_dataset_combined.json", "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    entries = [e for e in entries if all(k in e for k in ["text", "aspect", "sentiment"])]
    
    print(f"Loaded {len(entries)} entries")
    
    # Train aspect classification model (choose approach)
    print("\n" + "="*50)
    print("TRAINING ASPECT CLASSIFICATION MODEL")
    print("="*50)
    
    # Option 1: Multi-class classification (one aspect per text)
    aspect_classifier, classifier_tokenizer, aspect_to_id = train_aspect_classifier(entries, approach="multi_class")
    
    # Option 2: Multi-label classification (multiple aspects per text)
    # aspect_classifier, classifier_tokenizer, aspect_to_id = train_aspect_classifier(entries, approach="multi_label")
    
    # Train aspect term extraction model
    print("\n" + "="*50)
    print("TRAINING ASPECT TERM EXTRACTION MODEL")
    print("="*50)
    
    aspect_extractor, extractor_tokenizer = train_aspect_term_extractor(entries)
    
    # Example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    example_text = "The professor was excellent but the course material was too difficult"
    
    print(f"Text: {example_text}")
    print("\nAspect Classification:")
    predicted_aspects = predict_aspects_classification(
        example_text, aspect_classifier, classifier_tokenizer, aspect_to_id
    )
    for aspect, confidence in predicted_aspects:
        print(f"  {aspect}: {confidence:.3f}")
    
    print("\nAspect Term Extraction:")
    aspect_terms = extract_aspect_terms(example_text, aspect_extractor, extractor_tokenizer)
    print(f"  Extracted terms: {aspect_terms}")

if __name__ == "__main__":
    main()