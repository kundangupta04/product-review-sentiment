
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from src.features import load_data, split_data

MODEL_PATH = "models/sentiment_pipeline.joblib"
FIG_PATH = "outputs/figures/confusion_matrix.png"
REPORT_PATH = "outputs/reports/evaluation_report.txt"
LABELS = ["Negative", "Positive"]   # row/column order of the matrix
POSITIVE_CLASS = "Positive"         # the class treated as "positive" below


def evaluate():
    # 1. Rebuild the SAME split used in training (same file, same random_state)
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. Load the trained pipeline (no retraining) and predict on the test set
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    # 3. Overall metrics (Positive is the "positive" class)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=POSITIVE_CLASS)
    rec = recall_score(y_test, y_pred, pos_label=POSITIVE_CLASS)
    f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_CLASS)

    # 4. Confusion matrix: rows = actual, columns = predicted
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    tn, fp, fn, tp = cm.ravel()

    # 5. Per-class metrics for the Negative class (the minority class)
    neg_prec = precision_score(y_test, y_pred, pos_label="Negative")
    neg_rec = recall_score(y_test, y_pred, pos_label="Negative")
    neg_f1 = f1_score(y_test, y_pred, pos_label="Negative")

    # 6. Class balance in the test set and a "always say Positive" baseline
    counts = y_test.value_counts()
    baseline_acc = counts.max() / counts.sum()

    report = classification_report(y_test, y_pred, labels=LABELS, digits=4)

    # ---------------- Print results ----------------
    lines = []
    lines.append("=" * 60)
    lines.append(f"TEST SET SIZE: {len(y_test)} reviews")
    lines.append("")
    lines.append("OVERALL METRICS (Positive = the 'positive' class)")
    lines.append(f"Accuracy : {acc:.4f}")
    lines.append(f"Precision: {prec:.4f}")
    lines.append(f"Recall   : {rec:.4f}")
    lines.append(f"F1-score : {f1:.4f}")
    lines.append("")
    lines.append("NEGATIVE CLASS METRICS (minority class)")
    lines.append(f"Precision: {neg_prec:.4f}")
    lines.append(f"Recall   : {neg_rec:.4f}")
    lines.append(f"F1-score : {neg_f1:.4f}")
    lines.append("")
    lines.append("CONFUSION MATRIX (rows = actual, columns = predicted)")
    lines.append(f"                 Pred Negative   Pred Positive")
    lines.append(f"Actual Negative  {tn:>13}   {fp:>13}")
    lines.append(f"Actual Positive  {fn:>13}   {tp:>13}")
    lines.append("")
    lines.append(f"True Positives (TP) : {tp}")
    lines.append(f"True Negatives (TN) : {tn}")
    lines.append(f"False Positives (FP): {fp}")
    lines.append(f"False Negatives (FN): {fn}")
    lines.append("")
    lines.append("CLASSIFICATION REPORT")
    lines.append(report)
    lines.append("CLASS BALANCE CHECK (test set)")
    for label in LABELS:
        lines.append(f"{label}: {counts[label]} ({counts[label] / len(y_test):.2%})")
    lines.append(f"Accuracy of a dummy model that always says "
                 f"'{counts.idxmax()}': {baseline_acc:.4f}")
    lines.append("=" * 60)
    output = "\n".join(lines)
    print(output)

    # Sanity check: metrics computed by hand must match scikit-learn
    assert abs(prec - tp / (tp + fp)) < 1e-9
    assert abs(rec - tp / (tp + fn)) < 1e-9
    assert abs(acc - (tp + tn) / cm.sum()) < 1e-9
    print("Sanity check passed: hand-calculated values match scikit-learn.")

    # ---------------- Save report ----------------
    os.makedirs("outputs/reports", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Saved report to: {REPORT_PATH}")

    # ---------------- Save confusion matrix figure ----------------
    os.makedirs("outputs/figures", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, annot_kws={"size": 14})
    plt.title("Confusion Matrix - Logistic Regression (Test Set)")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {FIG_PATH}")


if __name__ == "__main__":
    evaluate()