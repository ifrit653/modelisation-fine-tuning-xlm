import json
import pandas as pd
from collections import Counter, defaultdict
import re

class AspectDatasetCleaner:
    def __init__(self):
        # Define consolidation rules for redundant aspects
        self.consolidation_rules = {
            # TEACHER aspects - consolidating similar/redundant ones
            "TEACHER#CLARITY": [
                "TEACHER#CLARITY",
                "TEACHER#EXPLANATION (sub-aspect of TEACHER#CLARITY)",
                "TEACHER#COMMUNICATION",
                "TEACHER#COMMUNICATION_SKILLS",
                "TEACHER#SPEAKING SKILL"
            ],
            
            "TEACHER#TEACHING_METHOD": [
                "TEACHER#TEACHING STYLE",
                "TEACHER#STYLE", 
                "TEACHER#TEACHING_STYLE",
                "TEACHER#TEACHING STYLES",
                "TEACHER#METHODS",
                "TEACHER#METHOD",
                "TEACHER#METHODOLOGY",
                "TEACHER#TECHNIQUE",
                "TEACHER#TECHNIQUES",
                "TEACHER#TEACHING TECHNIQUES",
                "TEACHER#TEACHING APPROACH",
                "TEACHER#APPROACH",
                "TEACHER#DELIVERY",
                "TEACHER#TEACHING SYSTEM",
                "TEACHER#STRATEGY"
            ],
            
            "TEACHER#SKILLS": [
                "TEACHER#EXPERTISE",
                "TEACHER#SKILL",
                "TEACHER#TEACHING SKILLS", 
                "TEACHER#ABILITY",
                "TEACHER#PERFORMANCE",
                "TEACHER#LECTURE SKILLS",
                "TEACHER#PRESENTATION_SKILLS",
                "TEACHER#MANAGEMENT_SKILLS",
                "TEACHER#EXPERIENCE"
            ],
            
            "TEACHER#ATTITUDE": [
                "TEACHER#ATTITUDE",
                "TEACHER#PERSONALITY",
                "TEACHER#CONFIDENCE",
                "TEACHER#DEDICATION",
                "TEACHER#EFFORT",
                "TEACHER#LEADERSHIP",
                "TEACHER#RESPONSIBILITY"
            ],
            
            "TEACHER#AVAILABILITY": [
                "TEACHER#AVAILABILITY",
                "TEACHER#HELPFULNESS",
                "TEACHER#COOPERATION",
                "TEACHER#INTERACTION",
                "TEACHER#GUIDANCE",
                "TEACHER#ATTENTION_TO_TOPICS",
                "TEACHER#ATTENTIVENESS"
            ],
            
            "TEACHER#ORGANIZATION": [
                "TEACHER#PUNCTUALITY",
                "TEACHER#ORGANISATION",
                "TEACHER#PROCESS"
            ],
            
            # COURSE aspects
            "COURSE#CONTENT": [
                "COURSE#CONTENT",
                "COURSE#QUALITY",
                "COURSE#CONcept",  # Typo in original
                "COURSE#BENEFIT"
            ],
            
            "COURSE#STRUCTURE": [
                "COURSE#STRUCTURE",
                "COURSE#SYSTEM",
                "COURSE#PROGRESS",
                "COURSE#DURATION",
                "COURSE#ORGANIZATION"  # If it exists
            ],
            
            "COURSE#RESOURCES": [
                "COURSE#RESOURCES",
                "TEACHER#RESOURCES",
                "COURSE#TEACHING METHOD"
            ],
            
            "COURSE#ENVIRONMENT": [
                "COURSE#ATMOSPHERE",
                "COURSE#OPPORTUNITY"
            ],
            
            # EVALUATION aspects
            "EVALUATION#ASSESSMENT": [
                "EVALUATION#FAIRNESS",
                "EVALUATION#DIFFICULTY",
                "EVALUATION#MARK DISTRIBUTION"
            ],
            
            "EVALUATION#FEEDBACK": [
                "EVALUATION#FEEDBACK",
                "TEACHER#FEEDBACK"
            ],
            
            # ENVIRONMENT aspects
            "ENVIRONMENT#FACILITIES": [
                "ENVIRONMENT#CLASSROOM",
                "ENVIRONMENT#RESOURCES",
                "ENVIRONMENT#QUALITY"
            ],
            
            # POLICIES
            "COURSE#POLICIES": [
                "TEACHER#POLICIES",
                "COURSE#POLICY",
                "COURSE#COMMUNICATION"
            ]
        }
        
        # Define aspects to remove (too vague or unclear)
        self.aspects_to_remove = [
            "TEACHER#IDENTITY",
            "TEACHER#LEARNING STYLE",  # This doesn't make sense
            "TEACHER#UNDERSTANDING",   # Too vague
            "TEACHER#PRACTICALITY"     # Unclear meaning
        ]
    
    def load_data(self, filepath):
        """Load the dataset"""
        print(f"Loading data from {filepath}...")
        
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            df = pd.read_csv(filepath)
            data = df.to_dict('records')
        
        print(f"Loaded {len(data)} entries")
        return data
    
    def analyze_current_aspects(self, data):
        """Analyze current aspect distribution"""
        aspects = [item['aspect'] for item in data if 'aspect' in item]
        aspect_counts = Counter(aspects)
        
        print(f"\nCurrent dataset has {len(aspect_counts)} unique aspects:")
        print("-" * 60)
        for aspect, count in aspect_counts.most_common():
            print(f"{aspect:<40} {count:>6}")
        
        return aspect_counts
    
    def create_consolidation_mapping(self):
        """Create a mapping from old aspects to new consolidated aspects"""
        mapping = {}
        
        for target_aspect, source_aspects in self.consolidation_rules.items():
            for source_aspect in source_aspects:
                mapping[source_aspect] = target_aspect
        
        # Add aspects to remove
        for aspect in self.aspects_to_remove:
            mapping[aspect] = None  # Mark for removal
        
        return mapping
    
    def apply_consolidation(self, data, consolidation_mapping):
        """Apply consolidation rules to the dataset"""
        consolidated_data = []
        removed_count = 0
        changed_count = 0
        
        for item in data:
            if 'aspect' not in item:
                continue
                
            original_aspect = item['aspect']
            
            if original_aspect in consolidation_mapping:
                new_aspect = consolidation_mapping[original_aspect]
                
                if new_aspect is None:
                    # Remove this entry
                    removed_count += 1
                    continue
                else:
                    # Update aspect
                    item = item.copy()  # Don't modify original
                    item['aspect'] = new_aspect
                    if original_aspect != new_aspect:
                        changed_count += 1
            
            consolidated_data.append(item)
        
        print(f"\nConsolidation results:")
        print(f"  - Entries removed: {removed_count}")
        print(f"  - Entries modified: {changed_count}")
        print(f"  - Final dataset size: {len(consolidated_data)}")
        
        return consolidated_data
    
    def analyze_consolidated_aspects(self, consolidated_data):
        """Analyze the consolidated aspect distribution"""
        aspects = [item['aspect'] for item in consolidated_data]
        aspect_counts = Counter(aspects)
        
        print(f"\nConsolidated dataset has {len(aspect_counts)} unique aspects:")
        print("-" * 60)
        
        total_samples = len(consolidated_data)
        for aspect, count in aspect_counts.most_common():
            percentage = (count / total_samples) * 100
            print(f"{aspect:<35} {count:>6} ({percentage:>5.1f}%)")
        
        return aspect_counts
    
    def check_minimum_samples(self, aspect_counts, min_samples=50):
        """Check which aspects have enough samples for training"""
        sufficient_aspects = []
        insufficient_aspects = []
        
        for aspect, count in aspect_counts.items():
            if count >= min_samples:
                sufficient_aspects.append((aspect, count))
            else:
                insufficient_aspects.append((aspect, count))
        
        print(f"\nAspects with >= {min_samples} samples: {len(sufficient_aspects)}")
        for aspect, count in sorted(sufficient_aspects, key=lambda x: x[1], reverse=True):
            print(f"  ✓ {aspect}: {count}")
        
        print(f"\nAspects with < {min_samples} samples: {len(insufficient_aspects)}")
        for aspect, count in sorted(insufficient_aspects, key=lambda x: x[1], reverse=True):
            print(f"  ✗ {aspect}: {count}")
        
        return sufficient_aspects, insufficient_aspects
    
    def suggest_further_consolidations(self, aspect_counts):
        """Suggest additional consolidations for aspects with few samples"""
        print("\n" + "="*60)
        print("SUGGESTED FURTHER CONSOLIDATIONS:")
        print("="*60)
        
        suggestions = {
            "TEACHER#GENERAL": [
                "TEACHER#ATTITUDE", 
                "TEACHER#ORGANIZATION"
            ],
            "COURSE#GENERAL": [
                "COURSE#STRUCTURE",
                "COURSE#ENVIRONMENT", 
                "COURSE#POLICIES"
            ],
            "ENVIRONMENT#GENERAL": [
                "ENVIRONMENT#FACILITIES"
            ]
        }
        
        for consolidated_name, aspects_to_merge in suggestions.items():
            total_count = sum(aspect_counts.get(asp, 0) for asp in aspects_to_merge)
            print(f"\n{consolidated_name}: {total_count} samples")
            for asp in aspects_to_merge:
                count = aspect_counts.get(asp, 0)
                print(f"  - {asp}: {count}")
    
    def create_balanced_dataset(self, consolidated_data, target_aspects=None, max_samples_per_aspect=1000):
        """Create a more balanced dataset by limiting samples per aspect"""
        if target_aspects is None:
            # Use aspects with sufficient samples
            aspect_counts = Counter(item['aspect'] for item in consolidated_data)
            target_aspects = [asp for asp, count in aspect_counts.items() if count >= 50]
        
        # Group data by aspect
        data_by_aspect = defaultdict(list)
        for item in consolidated_data:
            if item['aspect'] in target_aspects:
                data_by_aspect[item['aspect']].append(item)
        
        # Balance the dataset
        balanced_data = []
        for aspect, items in data_by_aspect.items():
            # Limit samples per aspect to reduce imbalance
            if len(items) > max_samples_per_aspect:
                # You might want to use stratified sampling here based on sentiment
                import random
                random.seed(42)
                items = random.sample(items, max_samples_per_aspect)
            
            balanced_data.extend(items)
        
        print(f"\nBalanced dataset created:")
        print(f"  - Total samples: {len(balanced_data)}")
        print(f"  - Aspects included: {len(target_aspects)}")
        
        # Show final distribution
        final_counts = Counter(item['aspect'] for item in balanced_data)
        for aspect, count in final_counts.most_common():
            print(f"  - {aspect}: {count}")
        
        return balanced_data, target_aspects
    
    def save_consolidated_data(self, consolidated_data, output_path):
        """Save the consolidated dataset"""
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        else:
            df = pd.DataFrame(consolidated_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\nConsolidated data saved to {output_path}")

def main():
    """Main function to clean and consolidate the aspect dataset"""
    
    # Initialize cleaner
    cleaner = AspectDatasetCleaner()
    
    # Load data
    input_file = "absa_dataset_combined.json"  # Adjust path as needed
    data = cleaner.load_data(input_file)
    
    # Analyze current aspects
    print("\n" + "="*70)
    print("CURRENT ASPECT ANALYSIS")
    print("="*70)
    original_counts = cleaner.analyze_current_aspects(data)
    
    # Create consolidation mapping
    consolidation_mapping = cleaner.create_consolidation_mapping()
    
    # Apply consolidation
    print("\n" + "="*70)
    print("APPLYING CONSOLIDATION RULES")
    print("="*70)
    consolidated_data = cleaner.apply_consolidation(data, consolidation_mapping)
    
    # Analyze consolidated results
    print("\n" + "="*70)
    print("CONSOLIDATED ASPECT ANALYSIS")
    print("="*70)
    consolidated_counts = cleaner.analyze_consolidated_aspects(consolidated_data)
    
    # Check minimum samples
    print("\n" + "="*70)
    print("SAMPLE SIZE ANALYSIS")
    print("="*70)
    sufficient, insufficient = cleaner.check_minimum_samples(consolidated_counts, min_samples=50)
    
    # Suggest further consolidations
    cleaner.suggest_further_consolidations(consolidated_counts)
    
    # Create balanced dataset
    print("\n" + "="*70)
    print("CREATING BALANCED DATASET")
    print("="*70)
    balanced_data, target_aspects = cleaner.create_balanced_dataset(
        consolidated_data, 
        max_samples_per_aspect=800  # Adjust this based on your needs
    )
    
    # Save consolidated data
    cleaner.save_consolidated_data(consolidated_data, "absa_dataset_consolidated.json")
    cleaner.save_consolidated_data(balanced_data, "absa_dataset_balanced.json")
    
    print("\n" + "="*70)
    print("CONSOLIDATION COMPLETE!")
    print("="*70)
    print(f"Original dataset: {len(data)} entries, {len(original_counts)} aspects")
    print(f"Consolidated dataset: {len(consolidated_data)} entries, {len(consolidated_counts)} aspects")
    print(f"Balanced dataset: {len(balanced_data)} entries, {len(target_aspects)} aspects")
    print("\nFiles created:")
    print("  - absa_dataset_consolidated.json (all consolidated data)")
    print("  - absa_dataset_balanced.json (balanced for training)")

if __name__ == "__main__":
    main()