import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
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

class ConsolidatedAspectDataset(Dataset):
    def __init__(self, texts, aspects, tokenizer, aspect2id, max_len=128):
        self.texts = texts
        self.aspects = aspects
        self.tokenizer = tokenizer
        self.aspect2id = aspect2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        aspect = self.aspects[idx]
        aspect_id = self.aspect2id[aspect]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(aspect_id, dtype=torch.long)
        }

class ConsolidatedAspectTrainer:
    def __init__(self, model_name="xlm-roberta-base", use_balanced_data=True):
        self.model_name = model_name
        self.use_balanced_data = use_balanced_data
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.aspect2id = {}
        self.id2aspect = {}
        self.class_weights = {}
        
    def load_consolidated_data(self):
        """Load the consolidated dataset"""
        if self.use_balanced_data:
            data_file = "absa_dataset_balanced.json"
            print("Loading balanced consolidated dataset...")
        else:
            data_file = "absa_dataset_consolidated.json"
            print("Loading full consolidated dataset...")
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} entries from {data_file}")
        except FileNotFoundError:
            print(f"Error: {data_file} not found!")
            print("Please run the consolidation script first to create the consolidated dataset.")
            raise
        
        return data
    
    def prepare_data(self, data):
        """Prepare texts and aspects from the loaded data"""
        texts = []
        aspects = []
        
        for item in data:
            if 'text' in item and 'aspect' in item:
                texts.append(item['text'])
                aspects.append(item['aspect'])
        
        print(f"Prepared {len(texts)} text-aspect pairs")
        return texts, aspects
    
    def analyze_aspect_distribution(self, aspects):
        """Analyze the distribution of aspects in the consolidated dataset"""
        aspect_counts = Counter(aspects)
        
        print(f"\nAspect distribution in consolidated dataset:")
        print("=" * 60)
        
        total_samples = len(aspects)
        for aspect, count in aspect_counts.most_common():
            percentage = (count / total_samples) * 100
            print(f"{aspect:<35} {count:>6} ({percentage:>5.1f}%)")
        
        # Create mappings
        self.aspect2id = {aspect: idx for idx, aspect in enumerate(aspect_counts.keys())}
        self.id2aspect = {idx: aspect for aspect, idx in self.aspect2id.items()}
        
        # Calculate class weights (less extreme than before due to consolidation)
        self.class_weights = {}
        for aspect, count in aspect_counts.items():
            aspect_id = self.aspect2id[aspect]
            # Use sqrt to make weights less extreme
            weight = np.sqrt(total_samples / (len(aspect_counts) * count))
            self.class_weights[aspect_id] = weight
        
        print(f"\nCreated mappings for {len(self.aspect2id)} consolidated aspects")
        return aspect_counts
    
    def create_datasets(self, texts, aspects, test_size=0.15, val_size=0.10):
        """Create train, validation, and test datasets"""
        # First split: separate test set
        train_val_texts, test_texts, train_val_aspects, test_aspects = train_test_split(
            texts, aspects, 
            test_size=test_size, 
            stratify=aspects, 
            random_state=42
        )
        
        # Second split: separate train and validation
        train_texts, val_texts, train_aspects, val_aspects = train_test_split(
            train_val_texts, train_val_aspects,
            test_size=val_size/(1-test_size),  # Adjust for the remaining data
            stratify=train_val_aspects,
            random_state=42
        )
        
        # Create datasets
        train_dataset = ConsolidatedAspectDataset(
            train_texts, train_aspects, self.tokenizer, self.aspect2id
        )
        val_dataset = ConsolidatedAspectDataset(
            val_texts, val_aspects, self.tokenizer, self.aspect2id
        )
        test_dataset = ConsolidatedAspectDataset(
            test_texts, test_aspects, self.tokenizer, self.aspect2id
        )
        
        print(f"\nDataset splits:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute comprehensive evaluation metrics"""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # Print detailed results during evaluation
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        conf_mat = confusion_matrix(labels, preds)
        print(conf_mat)
        
        # Classification Report
        print("\nDetailed Classification Report:")
        target_names = [self.id2aspect[i] for i in range(len(self.id2aspect))]
        report = classification_report(labels, preds, target_names=target_names, zero_division=0)
        print(report)
        
        return {
            "accuracy": accuracy,
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1
        }
    
    def create_weighted_trainer(self, model, training_args, train_dataset, val_dataset):
        """Create a trainer with class weights"""
        
        class WeightedAspectTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # Apply class weights
                device = logits.device
                weight_tensor = torch.tensor([
                    self.class_weights[i] for i in range(len(self.class_weights))
                ]).to(device).float()
                
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        return WeightedAspectTrainer(
            class_weights=self.class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
    
    def train_model(self, train_dataset, val_dataset, output_dir="./consolidated-aspect-model"):
        """Train the consolidated aspect extraction model"""
        
        print(f"\nInitializing model for {len(self.aspect2id)} consolidated aspects...")
        
        # Initialize model
        model = XLMRobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.aspect2id),
            id2label=self.id2aspect,
            label2id=self.aspect2id
        )
        
        # Training arguments - optimized for consolidated dataset
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./consolidated-aspect-logs",
            per_device_train_batch_size=32,  # Larger batch size now that we have fewer classes
            per_device_eval_batch_size=32,
            num_train_epochs=6,  # More epochs since we have cleaner data
            learning_rate=1e-5,  # Lower learning rate for fine-tuning
            weight_decay=0.01,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            report_to="none",
            logging_steps=50,
            save_total_limit=3,
            seed=42,
            dataloader_pin_memory=False,
            dataloader_num_workers=2,
        )
        
        # Create weighted trainer
        trainer = self.create_weighted_trainer(
            model, training_args, train_dataset, val_dataset
        )
        
        print("\nStarting training with consolidated aspects...")
        print("="*60)
        
        # Train the model
        trainer.train()
        
        # Save model and mappings
        print(f"\nSaving model to {output_dir}...")
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save consolidated aspect mappings
        mappings = {
            "aspect2id": self.aspect2id,
            "id2aspect": self.id2aspect,
            "class_weights": self.class_weights,
            "consolidation_info": {
                "total_aspects": len(self.aspect2id),
                "model_name": self.model_name,
                "training_completed": True,
                "use_balanced_data": self.use_balanced_data
            }
        }
        
        with open(f"{output_dir}/consolidated_aspect_mappings.json", "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        
        print(f"Model and mappings saved successfully!")
        
        return trainer
    
    def evaluate_on_test_set(self, trainer, test_dataset):
        """Evaluate the trained model on the test set"""
        print("\n" + "="*60)
        print("FINAL TEST SET EVALUATION")
        print("="*60)
        
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        
        print("\nFinal Test Results:")
        for metric, value in test_results.items():
            if isinstance(value, float) and 'eval_' in metric:
                print(f"  {metric}: {value:.4f}")
        
        return test_results
    
    def create_visualization(self, aspect_counts, output_dir="./consolidated-aspect-model"):
        """Create visualization of aspect distribution"""
        plt.figure(figsize=(15, 8))
        
        aspects = list(aspect_counts.keys())
        counts = list(aspect_counts.values())
        
        # Create bar plot
        bars = plt.bar(range(len(aspects)), counts)
        plt.xlabel('Aspects')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Consolidated Aspects')
        plt.xticks(range(len(aspects)), [asp.replace('_', '\n') for asp in aspects], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/aspect_distribution.png", dpi=300, bbox_inches='tight')
        print(f"Aspect distribution plot saved to {output_dir}/aspect_distribution.png")
        plt.close()

def main():
    """Main training function for consolidated aspects"""
    
    print("="*80)
    print("TRAINING CONSOLIDATED ASPECT EXTRACTION MODEL")
    print("="*80)
    
    # Initialize trainer
    trainer = ConsolidatedAspectTrainer(use_balanced_data=True)
    
    # Load consolidated data
    try:
        data = trainer.load_consolidated_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Prepare data
    texts, aspects = trainer.prepare_data(data)
    
    # Analyze aspect distribution
    aspect_counts = trainer.analyze_aspect_distribution(aspects)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(texts, aspects)
    
    # Train model
    model_trainer = trainer.train_model(train_dataset, val_dataset)
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test_set(model_trainer, test_dataset)
    
    # Create visualization
    trainer.create_visualization(aspect_counts)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"✅ Model trained on {len(trainer.aspect2id)} consolidated aspects")
    print(f"✅ Final test accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    print(f"✅ Final test macro F1: {test_results.get('eval_f1_macro', 0):.4f}")
    print(f"✅ Model saved to: ./consolidated-aspect-model/")
    print(f"✅ Mappings saved to: ./consolidated-aspect-model/consolidated_aspect_mappings.json")
    
    print("\nNext steps:")
    print("1. Test the model with the inference script")
    print("2. Integrate with your sentiment analysis model")
    print("3. Deploy the complete ABSA system")

if __name__ == "__main__":
    main()