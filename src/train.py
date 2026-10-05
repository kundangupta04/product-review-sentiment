import os

import joblib
from sklearn.linear_model import LogisticRegression

from src.features import build_tfidf_pipeline, load_data, split_data

MODEL_PATH = "models/sentiment_pipeline.joblib"
RANDOM_STATE = 42


def build_model_pipeline():
    """TF-IDF step (from Step 5) + Logistic Regression as the final step."""
    pipe = build_tfidf_pipeline()
    pipe.steps.append((
        "classifier",
        LogisticRegression(
            max_iter=1000,              # enough iterations to converge
            random_state=RANDOM_STATE,  # reproducible results
            class_weight="balanced",    # compensates for fewer Negative reviews
        ),
    ))
    return pipe


def train_and_save():
    """Split, train on the training set only, save the pipeline, return test data."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model_pipeline()
    model.fit(X_train, y_train)  # TF-IDF and classifier both learn from TRAIN only

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained pipeline to: {MODEL_PATH}")

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = train_and_save()

    print("=" * 60)
    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))
    print("Classes learned :", list(model.classes_))

    # ---- Predict on the test set ----
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS FROM THE TEST SET\n")
    for i in range(10):
        print(f"Review   : {X_test.iloc[i][:90]}...")
        print(f"Actual   : {y_test.iloc[i]}")
        print(f"Predicted: {y_pred[i]}")
        print("Correct  :", "yes" if y_test.iloc[i] == y_pred[i] else "NO")
        print("-" * 40)

    # ---- Full-flow test: raw review -> cleaning -> TF-IDF -> LR -> prediction ----
    from src.preprocessing import clean_text

    print("=" * 60)
    print("FULL FLOW TEST (raw text -> prediction)\n")
    loaded = joblib.load(MODEL_PATH)  # also proves the saved file loads
    samples = [
        "The product quality is excellent and I am very happy with it!",
        "It stopped working after two days. Very disappointed.",
        "This is <b>not</b> good at all.",
    ]
    for raw in samples:
        cleaned = clean_text(raw)
        label = loaded.predict([cleaned])[0]
        print(f"Raw     : {raw}")
        print(f"Cleaned : {cleaned}")
        print(f"Result  : {label}\n")