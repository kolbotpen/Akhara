import csv

rows = []

with open("dataset_generation/translated_nouns.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append({"id": row["id"], "khmer_text": row["translation"]})

with open("dataset_generation/khmer_name_corpus.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append({"id": row["id"], "khmer_text": row["khmer_name"]})

with open("dataset_generation/merged_corpus.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "khmer_text"])
    writer.writeheader()
    writer.writerows(rows)
    
print(f"Done. {len(rows)} rows written to dataset_generation/merged_corpus.csv.csv")