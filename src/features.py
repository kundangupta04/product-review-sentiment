
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------- SETTINGS ----------------
DATA_PATH = "data/processed/reviews_cleaned.csv"
TEST_SIZE = 0.2          # 80% train, 20% test
RANDOM_STATE = 42        # fixed seed -> same split every run
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)     # single words AND two-word phrases like "not good"
# ------------------------------------------

# scikit-learn's English stopword list contains negation words.
# Removing them would turn "not good" into "good", so we keep them.
KEEP_WORDS = {
    "not", "no", "nor", "never", "cannot", "none", "nobody",
    "nothing", "neither", "nowhere", "without",
}
CUSTOM_STOP_WORDS = sorted(ENGLISH_STOP_WORDS - KEEP_WORDS)


def load_data(path=DATA_PATH):
    """Load the cleaned dataset and return X (text) and y (label)."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["clean_review", "sentiment"])
    return df["clean_review"], df["sentiment"]


def split_data(X, y):
    """80/20 split, stratified so both sets keep the same class ratio."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_tfidf_pipeline():
    """A scikit-learn Pipeline holding the TF-IDF step.
    In Step 6 we add Logistic Regression as a second step of this pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=MAX_FEATURES,
            stop_words=CUSTOM_STOP_WORDS,
            ngram_range=NGRAM_RANGE,
        )),
    ])


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ---- Fit on TRAINING data only, then transform both ----
    pipe = build_tfidf_pipeline()
    X_train_tfidf = pipe.fit_transform(X_train)   # learns vocabulary + IDF from train
    X_test_tfidf = pipe.transform(X_test)         # only applies what train learned

    print("=" * 60)
    print("SPLIT SIZES")
    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    print("\nTF-IDF MATRIX SHAPES (rows = reviews, columns = features)")
    print("Training matrix :", X_train_tfidf.shape)
    print("Testing matrix  :", X_test_tfidf.shape)

    print("\n" + "=" * 60)
    print("CHECK 1: class ratio preserved (stratification)")
    ratios = pd.DataFrame({
        "full": y.value_counts(normalize=True),
        "train": y_train.value_counts(normalize=True),
        "test": y_test.value_counts(normalize=True),
    }).round(4)
    print(ratios)

    print("\nCHECK 2: no review appears in both train and test")
    overlap = set(X_train.index) & set(X_test.index)
    print("Overlapping rows:", len(overlap))
    assert len(overlap) == 0

    print("\nCHECK 3: vocabulary comes from training data only")
    vocab = pipe.named_steps["tfidf"].vocabulary_
    vocab_words = {w for term in vocab for w in term.split()}
    train_words = set(" ".join(X_train).split())
    test_only_words = set(" ".join(X_test).split()) - train_words
    leaked = vocab_words & test_only_words
    print("Words that appear only in test data:", len(test_only_words))
    print("Of those, words found in the vocabulary:", len(leaked))
    assert len(leaked) == 0

    print("\nCHECK 4: negation information is preserved")
    for term in ["not", "no", "never"]:
        print(f"  '{term}' in vocabulary:", term in vocab)
    bigrams = [t for t in vocab if t.startswith("not ")][:8]
    print("  Sample 'not ...' phrases:", bigrams)
    assert "not" in vocab

    print("\nSample features:",
          list(pipe.named_steps["tfidf"].get_feature_names_out()[:10]))
    density = X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])
    print(f"Non-zero cells in training matrix: {density:.2%} (rest are zeros)")

    print("\nSTEP 5 TEST PASSED")