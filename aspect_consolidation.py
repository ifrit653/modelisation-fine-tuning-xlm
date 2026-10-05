import json
import pandas as pd
from collections import Counter, defaultdict
import re
import difflib

class AspectDatasetCleaner:
    def __init__(self):
        # Consolidation rules (manual grouping of redundant aspects)
        self.consolidation_rules = {
            "TEACHER#CLARITY": [
                "TEACHER#CLARITY", "TEACHER#COMMUNICATION", "TEACHER#COMMUNICATION_SKILLS",
                "TEACHER#SPEAKING SKILL", "TEACHER#EXPLANATION"
            ],
            "TEACHER#TEACHING_METHOD": [
                "TEACHER#TEACHING STYLE", "TEACHER#STYLE", "TEACHER#METHOD",
                "TEACHER#METHODOLOGY", "TEACHER#TECHNIQUE", "TEACHER#DELIVERY",
                "TEACHER#STRATEGY"
            ],
            "TEACHER#SKILLS": [
                "TEACHER#EXPERTISE", "TEACHER#SKILL", "TEACHER#ABILITY",
                "TEACHER#PERFORMANCE", "TEACHER#EXPERIENCE"
            ],
            "TEACHER#ATTITUDE": [
                "TEACHER#ATTITUDE", "TEACHER#PERSONALITY", "TEACHER#DEDICATION",
                "TEACHER#EFFORT", "TEACHER#CONFIDENCE"
            ],
            "TEACHER#AVAILABILITY": [
                "TEACHER#AVAILABILITY", "TEACHER#HELPFULNESS", "TEACHER#COOPERATION",
                "TEACHER#INTERACTION", "TEACHER#GUIDANCE"
            ],
            "COURSE#CONTENT": [
                "COURSE#CONTENT", "COURSE#QUALITY", "COURSE#BENEFIT", "COURSE#CONcept"
            ],
            "COURSE#STRUCTURE": [
                "COURSE#STRUCTURE", "COURSE#SYSTEM", "COURSE#PROGRESS",
                "COURSE#DURATION", "COURSE#ORGANIZATION"
            ],
            "EVALUATION#ASSESSMENT": [
                "EVALUATION#FAIRNESS", "EVALUATION#DIFFICULTY", "EVALUATION#MARK DISTRIBUTION"
            ],
            "EVALUATION#FEEDBACK": [
                "EVALUATION#FEEDBACK", "TEACHER#FEEDBACK"
            ],
            "ENVIRONMENT#FACILITIES": [
                "ENVIRONMENT#CLASSROOM", "ENVIRONMENT#RESOURCES", "ENVIRONMENT#QUALITY"
            ]
        }

        # Aspects too vague to keep
        self.aspects_to_remove = [
            "TEACHER#IDENTITY", "TEACHER#LEARNING STYLE", 
            "TEACHER#UNDERSTANDING", "TEACHER#PRACTICALITY"
        ]

    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
        return data

    def analyze_current_aspects(self, data):
        aspects = [item['aspect'] for item in data if 'aspect' in item]
        aspect_counts = Counter(aspects)
        print(f"\nCurrent dataset has {len(aspect_counts)} unique aspects:")
        for aspect, count in aspect_counts.most_common():
            print(f"{aspect:<40} {count:>6}")
        return aspect_counts

    def create_consolidation_mapping(self, aspect_counts):
        mapping = {}

        # Add manual rules
        for target, sources in self.consolidation_rules.items():
            for s in sources:
                mapping[s] = target

        # Remove vague aspects
        for asp in self.aspects_to_remove:
            mapping[asp] = None

        # Auto-detect near-duplicates (Levenshtein/difflib)
        all_aspects = list(aspect_counts.keys())
        for asp in all_aspects:
            if asp not in mapping:
                # Find closest existing aspect in mapping
                close = difflib.get_close_matches(asp, mapping.keys(), n=1, cutoff=0.85)
                if close:
                    mapping[asp] = mapping[close[0]]
        return mapping

    def apply_consolidation(self, data, mapping):
        consolidated = []
        removed, changed = 0, 0
        for item in data:
            asp = item['aspect']
            if asp in mapping:
                new_asp = mapping[asp]
                if new_asp is None:
                    removed += 1
                    continue
                if asp != new_asp:
                    item = item.copy()
                    item['aspect'] = new_asp
                    changed += 1
            consolidated.append(item)
        print(f"\nConsolidation results:")
        print(f"  - Removed: {removed}")
        print(f"  - Modified: {changed}")
        print(f"  - Final dataset size: {len(consolidated)}")
        return consolidated

    def save(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved dataset to {path}")

def main():
    cleaner = AspectDatasetCleaner()
    data = cleaner.load_data("aspect-dataset.json")

    counts = cleaner.analyze_current_aspects(data)
    mapping = cleaner.create_consolidation_mapping(counts)
    consolidated = cleaner.apply_consolidation(data, mapping)

    # Save both versions
    cleaner.save(consolidated, "absa_dataset_consolidated.json")

if __name__ == "__main__":
    main()
