import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import json
import random

# Set seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class SimpleAspectDataset(Dataset):
    def __init__(self, texts, aspects, tokenizer, aspect2id, max_len=128):
        self.texts = texts
        self.aspects = aspects
        self.tokenizer = tokenizer
        self.aspect2id = aspect2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
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

class PracticalAspectTrainer:
    def __init__(self, top_k_aspects=6):
        """
        Simple, practical trainer focusing on top aspects only
        
        Args:
            top_k_aspects: Number of top aspects to keep (default: 6)
        """
        self.top_k_aspects = top_k_aspects
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.aspect2id = {}
        self.id2aspect = {}
        
    def load_and_prepare_data(self):
        """Load data and keep only top K aspects"""
        print("Loading dataset...")
        
        # Load data
        for filename in ["absa_dataset_balanced.json", "absa_dataset_consolidated.json"]:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Loaded {len(data)} entries from {filename}")
                break
            except FileNotFoundError:
                continue
        
        # Extract texts and aspects
        texts = []
        aspects = []
        
        for item in data:
            if 'text' in item and 'aspect' in item and item['text'].strip():
                texts.append(item['text'].strip())
                aspects.append(item['aspect'])
        
        # Count aspects
        aspect_counts = Counter(aspects)
        print(f"\nOriginal aspect distribution:")
        for aspect, count in aspect_counts.most_common():
            print(f"  {aspect}: {count}")
        
        # Keep only top K aspects
        top_aspects = [asp for asp, _ in aspect_counts.most_common(self.top_k_aspects)]
        
        print(f"\n✅ Keeping top {self.top_k_aspects} aspects:")
        filtered_texts = []
        filtered_aspects = []
        
        for text, aspect in zip(texts, aspects):
            if aspect in top_aspects:
                filtered_texts.append(text)
                filtered_aspects.append(aspect)
        
        # Show final distribution
        final_counts = Counter(filtered_aspects)
        for aspect in top_aspects:
            count = final_counts[aspect]
            print(f"  {aspect}: {count}")
        
        print(f"\nTotal samples after filtering: {len(filtered_texts)}")
        
        return filtered_texts, filtered_aspects
    
    def balance_data_simple(self, texts, aspects):
        """Simple oversampling to balance classes"""
        aspect_counts = Counter(aspects)
        max_count = max(aspect_counts.values())
        target_count = min(max_count, 400)  # Cap at 400 per class
        
        print(f"\nBalancing to {target_count} samples per class...")
        
        # Group by aspect
        aspect_groups = {}
        for text, aspect in zip(texts, aspects):
            if aspect not in aspect_groups:
                aspect_groups[aspect] = []
            aspect_groups[aspect].append(text)
        
        # Oversample minority classes
        balanced_texts = []
        balanced_aspects = []
        
        for aspect, texts_list in aspect_groups.items():
            current_count = len(texts_list)
            
            if current_count < target_count:
                # Oversample
                multiplier = target_count // current_count
                remainder = target_count % current_count
                
                balanced_texts.extend(texts_list * multiplier)
                balanced_texts.extend(random.sample(texts_list, remainder))
                balanced_aspects.extend([aspect] * target_count)
            else:
                # Undersample or keep as is
                sampled = random.sample(texts_list, min(target_count, current_count))
                balanced_texts.extend(sampled)
                balanced_aspects.extend([aspect] * len(sampled))
        
        # Shuffle
        combined = list(zip(balanced_texts, balanced_aspects))
        random.shuffle(combined)
        balanced_texts, balanced_aspects = zip(*combined)
        
        print(f"Balanced dataset size: {len(balanced_texts)}")
        final_counts = Counter(balanced_aspects)
        for aspect, count in final_counts.most_common():
            print(f"  {aspect}: {count}")
        
        return list(balanced_texts), list(balanced_aspects)
    
    def setup_mappings(self, aspects):
        """Create aspect mappings"""
        unique_aspects = sorted(set(aspects))
        self.aspect2id = {asp: idx for idx, asp in enumerate(unique_aspects)}
        self.id2aspect = {idx: asp for asp, idx in self.aspect2id.items()}
        
        print(f"\n✅ Created mappings for {len(self.aspect2id)} aspects")
    
    def create_datasets(self, texts, aspects):
        """Create train/val/test splits"""
        # First split: test set
        train_val_texts, test_texts, train_val_aspects, test_aspects = train_test_split(
            texts, aspects, test_size=0.15, stratify=aspects, random_state=42
        )
        
        # Second split: train and validation
        train_texts, val_texts, train_aspects, val_aspects = train_test_split(
            train_val_texts, train_val_aspects, test_size=0.12, 
            stratify=train_val_aspects, random_state=42
        )
        
        # Create datasets
        train_dataset = SimpleAspectDataset(
            train_texts, train_aspects, self.tokenizer, self.aspect2id
        )
        val_dataset = SimpleAspectDataset(
            val_texts, val_aspects, self.tokenizer, self.aspect2id
        )
        test_dataset = SimpleAspectDataset(
            test_texts, test_aspects, self.tokenizer, self.aspect2id
        )
        
        print(f"\nDataset splits:")
        print(f"  Training: {len(train_dataset)}")
        print(f"  Validation: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        
        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        
        # Classification report
        target_names = [self.id2aspect[i] for i in range(len(self.id2aspect))]
        report = classification_report(labels, preds, target_names=target_names, zero_division=0)
        print(f"\n{report}")
        
        return {
            "accuracy": accuracy,
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1
        }
    
    def train_model(self, train_dataset, val_dataset):
        """Train with simple, proven settings"""
        
        print(f"\n🚀 Training model for {len(self.aspect2id)} aspects...")
        
        # Simple model - no excessive dropout
        model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(self.aspect2id),
            id2label=self.id2aspect,
            label2id=self.aspect2id
        )
        
        # Simple, proven training arguments
        training_args = TrainingArguments(
            output_dir="./practical-aspect-model",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./practical-logs",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=4,  # Fewer epochs
            learning_rate=2e-5,  # Standard learning rate
            weight_decay=0.01,   # Standard weight decay
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            report_to="none",
            logging_steps=50,
            save_total_limit=2,
            seed=42
        )
        
        # Standard trainer - no fancy loss functions
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        print("Training started...")
        trainer.train()
        
        # Save model
        model.save_pretrained("./practical-aspect-model")
        self.tokenizer.save_pretrained("./practical-aspect-model")
        
        # Save mappings
        mappings = {
            "aspect2id": self.aspect2id,
            "id2aspect": self.id2aspect,
            "num_aspects": len(self.aspect2id),
            "model_info": {
                "type": "practical_simple",
                "top_k_aspects": self.top_k_aspects
            }
        }
        
        with open("./practical-aspect-model/practical_mappings.json", "w") as f:
            json.dump(mappings, f, indent=2)
        
        print("✅ Model saved to ./practical-aspect-model/")
        
        return trainer

def main():
    """Main training function with practical approach"""
    
    print("="*80)
    print("PRACTICAL ASPECT CLASSIFICATION - TOP 6 ASPECTS")
    print("="*80)
    print("\n🎯 Strategy: Focus on top aspects with good data distribution")
    print("   This approach prioritizes reliability over coverage.\n")
    
    # Initialize trainer (try top 6 aspects first)
    trainer = PracticalAspectTrainer(top_k_aspects=6)
    
    # Load and prepare data
    texts, aspects = trainer.load_and_prepare_data()
    
    # Balance data
    texts, aspects = trainer.balance_data_simple(texts, aspects)
    
    # Setup mappings
    trainer.setup_mappings(aspects)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(texts, aspects)
    
    # Train model
    model_trainer = trainer.train_model(train_dataset, val_dataset)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    test_results = model_trainer.evaluate(eval_dataset=test_dataset)
    
    print(f"\n📊 Final Results:")
    print(f"   Accuracy: {test_results.get('eval_accuracy', 0):.2%}")
    print(f"   Macro F1: {test_results.get('eval_f1_macro', 0):.2%}")
    print(f"   Weighted F1: {test_results.get('eval_f1_weighted', 0):.2%}")
    
    # Verdict
    if test_results.get('eval_accuracy', 0) > 0.70:
        print("\n🎉 EXCELLENT! Model is production-ready!")
    elif test_results.get('eval_accuracy', 0) > 0.60:
        print("\n✅ GOOD! Model performance is acceptable.")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT. Try with top 4-5 aspects only.")
    
    print("\n💡 Next steps:")
    print("   1. If accuracy > 70%: Deploy this model")
    print("   2. If accuracy 60-70%: Acceptable, can use with human review")
    print("   3. If accuracy < 60%: Reduce to top 4-5 aspects and retrain")
    
    return test_results

if __name__ == "__main__":
    results = main()