import os
import pandas as pd
from src.preprocessing import clean_text

INPUT_PATH = "data/processed/reviews_labeled.csv"
OUTPUT_PATH = "data/processed/reviews_cleaned.csv"

df = pd.read_csv(INPUT_PATH)
print("Loaded dataset:", df.shape)

# ---- 1. Handle missing values (before cleaning) ----
missing_before = df["review"].isnull().sum()
df = df.dropna(subset=["review", "sentiment"])
print(f"Missing reviews dropped: {missing_before}")

# ---- 2. Apply the cleaning function ----
df["clean_review"] = df["review"].apply(clean_text)

# ---- 3. Remove reviews that became empty after cleaning ----
empty_count = (df["clean_review"].str.strip() == "").sum()
df = df[df["clean_review"].str.strip() != ""]
print(f"Empty reviews after cleaning (dropped): {empty_count}")

# ---- 4. Remove duplicate reviews (based on cleaned text) ----
dup_count = df["clean_review"].duplicated().sum()
df = df.drop_duplicates(subset="clean_review", keep="first")
print(f"Duplicate reviews dropped: {dup_count}")

df = df.reset_index(drop=True)

# ---- 5. Show examples: Original -> Cleaned ----
print("\n" + "=" * 60)
print("EXAMPLES: ORIGINAL -> CLEANED")
for i in range(5):
    print(f"\n[{df.loc[i, 'sentiment']}]")
    print("ORIGINAL:", df.loc[i, "review"][:200])
    print("CLEANED :", df.loc[i, "clean_review"][:200])

# ---- 6. Final checks ----
print("\n" + "=" * 60)
print("FINAL CHECKS")
print("Final shape:", df.shape)
print("Missing values after cleaning:\n", df.isnull().sum())
print("Empty reviews remaining:", (df["clean_review"].str.strip() == "").sum())
print("\nSentiment counts:")
print(df["sentiment"].value_counts())

# ---- 7. Save ----
os.makedirs("data/processed", exist_ok=True)
df[["review", "clean_review", "sentiment"]].to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")