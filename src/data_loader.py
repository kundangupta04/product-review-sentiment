
import os
import pandas as pd

# ---------------- SETTINGS (change only these) ----------------
RAW_PATH = "data/raw/Reviews.csv"      # or "data/raw/amazon_cells_labelled.txt"
PROCESSED_PATH = "data/processed/reviews_labeled.csv"
SAMPLE_SIZE = 20000                    # Kaggle file is huge, so we use a sample
RANDOM_STATE = 42                      # fixed seed -> same sample every run
# --------------------------------------------------------------


def load_and_label(path=RAW_PATH, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE):
    """Read the raw file and return a DataFrame with columns: review, sentiment."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'. Put it in data/raw/ first.")

    if path.endswith(".txt"):
        # UCI format: "sentence <TAB> label", no header row, label is 0 or 1
        df = pd.read_csv(path, sep="\t", header=None, names=["review", "label"])
        df["sentiment"] = df["label"].map({1: "Positive", 0: "Negative"})
        return df[["review", "sentiment"]]

    # Kaggle Amazon Fine Food Reviews format: columns "Text" and "Score" (1-5 stars)
    df = pd.read_csv(path, usecols=["Text", "Score"])
    df = df.rename(columns={"Text": "review"})

    df = df[df["Score"] != 3]  # 3 stars are neutral, so we drop them (binary task)
    df["sentiment"] = df["Score"].apply(lambda s: "Positive" if s >= 4 else "Negative")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    return df[["review", "sentiment"]].reset_index(drop=True)


def inspect_dataset(df):
    """Print the basic information required for Step 2."""
    print("=" * 60)
    print("FIRST 5 RECORDS")
    print(df.head())

    print("\n" + "=" * 60)
    print("SHAPE (rows, columns):", df.shape)

    print("\nCOLUMN NAMES:", list(df.columns))

    print("\n" + "=" * 60)
    print("MISSING VALUES PER COLUMN")
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS (same review and sentiment):", df.duplicated().sum())
    print("DUPLICATE REVIEW TEXTS (same text only):   ", df["review"].duplicated().sum())

    print("\n" + "=" * 60)
    print("SENTIMENT CLASS DISTRIBUTION (count)")
    print(df["sentiment"].value_counts())
    print("\nSENTIMENT CLASS DISTRIBUTION (percentage)")
    print((df["sentiment"].value_counts(normalize=True) * 100).round(2))


if __name__ == "__main__":
    data = load_and_label()
    inspect_dataset(data)

    os.makedirs("data/processed", exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved labeled dataset to: {PROCESSED_PATH}")