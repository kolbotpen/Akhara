import csv

rows = []
max_translated_id = 0

with open("dataset_generation/translated_nouns.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append({"id": row["id"], "khmer_text": row["translation"]})
        max_translated_id = max(max_translated_id, int(row["id"]))

with open("dataset_generation/khmer_name_corpus.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        new_id = int(row["id"]) + max_translated_id
        rows.append({"id": str(new_id), "khmer_text": row["khmer_name"]})

with open("dataset_generation/merged_corpus.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "khmer_text"])
    writer.writeheader()
    writer.writerows(rows)
    
print(f"Done. {len(rows)} rows written to dataset_generation/merged_corpus.csv")