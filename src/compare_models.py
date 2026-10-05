
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.naive_bayes import MultinomialNB

from src.features import build_tfidf_pipeline, load_data, split_data

LR_MODEL_PATH = "models/sentiment_pipeline.joblib"
TABLE_PATH = "outputs/reports/model_comparison.csv"


def build_nb_pipeline():
    """Same TF-IDF step as Step 5, with Naive Bayes as the final step."""
    pipe = build_tfidf_pipeline()
    pipe.steps.append(("classifier", MultinomialNB()))
    return pipe


def score_model(name, y_test, y_pred):
    """Return one table row of real metrics computed from test predictions."""
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Pos)": precision_score(y_test, y_pred, pos_label="Positive"),
        "Recall (Pos)": recall_score(y_test, y_pred, pos_label="Positive"),
        "F1 (Pos)": f1_score(y_test, y_pred, pos_label="Positive"),
        "Precision (Neg)": precision_score(y_test, y_pred, pos_label="Negative"),
        "Recall (Neg)": recall_score(y_test, y_pred, pos_label="Negative"),
        "F1 (Neg)": f1_score(y_test, y_pred, pos_label="Negative"),
        "Macro F1": f1_score(y_test, y_pred, average="macro"),
    }


if __name__ == "__main__":
    # Same data, same split as every earlier step
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Logistic Regression: load the saved model (no retraining, no changes)
    lr_model = joblib.load(LR_MODEL_PATH)
    lr_pred = lr_model.predict(X_test)

    # Naive Bayes: TF-IDF and classifier both fitted on TRAINING data only
    nb_model = build_nb_pipeline()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)

    table = pd.DataFrame([
        score_model("Logistic Regression", y_test, lr_pred),
        score_model("Multinomial Naive Bayes", y_test, nb_pred),
    ]).set_index("Model").round(4)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("=" * 60)
    print(f"COMPARISON ON THE SAME TEST SET ({len(y_test)} reviews)")
    print(table.T)   # transposed so it fits on screen

    # Naive Bayes confusion counts (rows = actual, columns = predicted)
    for name, pred in [("Logistic Regression", lr_pred), ("Naive Bayes", nb_pred)]:
        neg_as_neg = ((y_test == "Negative") & (pred == "Negative")).sum()
        neg_as_pos = ((y_test == "Negative") & (pred == "Positive")).sum()
        pos_as_neg = ((y_test == "Positive") & (pred == "Negative")).sum()
        pos_as_pos = ((y_test == "Positive") & (pred == "Positive")).sum()
        print(f"\n{name}: TN={neg_as_neg}  FP={neg_as_pos}  FN={pos_as_neg}  TP={pos_as_pos}")

    os.makedirs("outputs/reports", exist_ok=True)
    table.to_csv(TABLE_PATH)
    print(f"\nSaved comparison table to: {TABLE_PATH}")