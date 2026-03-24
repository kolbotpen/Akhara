import pandas as pd

df = pd.read_csv('generated_nouns.csv')
df = df.drop_duplicates(subset=['noun'])
df.to_csv("generated_nouns.csv", index=False)