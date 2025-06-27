import json
import pandas as pd

# Load both datasets
with open("./absa_dataset_en_cleaned.json", "r", encoding="utf-8") as f:
    en_data = json.load(f)

with open("./absa_dataset_fr_cleaned.json", "r", encoding="utf-8") as f:
    fr_data = json.load(f)

# Merge the lists
merged_data = en_data + fr_data

# (Optional) Shuffle to mix languages
import random
random.shuffle(merged_data)

# Save merged version
with open("./absa_dataset_combined.json", "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)
