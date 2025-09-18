import json
import pandas as pd

# Load both datasets
with open("./data/absa_dataset.json", "r", encoding="utf-8") as f:
    en_data = json.load(f)

with open("./data/absa_dataset_fr.json", "r", encoding="utf-8") as f:
    fr_data = json.load(f)

# Merge the lists
merged_data = en_data + fr_data

# (Optional) Shuffle to mix languages
import random
random.shuffle(merged_data)

# Save merged version
with open("./aspect-dataset.json", "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)
