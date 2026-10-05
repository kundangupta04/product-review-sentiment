import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.getcwd())
APP_PATH = os.path.join(os.getcwd(), "app.py")   # absolute path to the Streamlit app

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)

from src.features import build_tfidf_pipeline, load_data, split_data
from src.predict import load_model, predict_sentiment
from src.preprocessing import clean_text
from src.train import MODEL_PATH, build_model_pipeline

results = []


def check(number, name, condition, detail=""):
    """Record and print one PASS/FAIL line."""
    status = "PASS" if condition else "FAIL"
    results.append(condition)
    print(f"[{status}] {number}. {name}" + (f"  ({detail})" if detail else ""))


print("=" * 60)
print("END-TO-END PROJECT TEST")
print("=" * 60)

# ---- 1. Dataset loads correctly ----
raw_exists = os.path.exists("data/raw/Reviews.csv") or os.path.exists(
    "data/raw/amazon_cells_labelled.txt")
labeled = pd.read_csv("data/processed/reviews_labeled.csv")
check(1, "Dataset loads correctly",
      raw_exists and list(labeled.columns) == ["review", "sentiment"]
      and set(labeled["sentiment"]) == {"Positive", "Negative"},
      f"{labeled.shape[0]} rows, classes: {sorted(labeled['sentiment'].unique())}")

# ---- 2. Cleaning works ----
samples = {
    "<b>Great</b> product!!! http://x.com": "great product",
    "It doesn't work": "it does not work",
    "Never again!": "never again",
}
check(2, "Cleaning works (HTML, URLs, punctuation, negation kept)",
      all(clean_text(raw) == exp for raw, exp in samples.items()))

# ---- 3. No unexpected missing values ----
cleaned = pd.read_csv("data/processed/reviews_cleaned.csv")
missing = int(cleaned[["clean_review", "sentiment"]].isnull().sum().sum())
empty = int((cleaned["clean_review"].str.strip() == "").sum())
dups = int(cleaned["clean_review"].duplicated().sum())
check(3, "No missing, empty or duplicate reviews remain",
      missing == 0 and empty == 0 and dups == 0,
      f"missing={missing}, empty={empty}, duplicates={dups}")

# ---- 4. Train/test split works ----
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
ratio_gap = abs(y_train.value_counts(normalize=True)["Negative"]
                - y_test.value_counts(normalize=True)["Negative"])
check(4, "Train/test split works (80/20, stratified, no overlap)",
      len(X_train) + len(X_test) == len(X)
      and abs(len(X_test) / len(X) - 0.2) < 0.01
      and ratio_gap < 0.005
      and len(set(X_train.index) & set(X_test.index)) == 0,
      f"train={len(X_train)}, test={len(X_test)}")

# ---- 5. TF-IDF fitted only on training data ----
fresh_tfidf = build_tfidf_pipeline()
fresh_tfidf.fit(X_train)                        # fit on TRAIN only
train_words = set(" ".join(X_train).split())
test_only = set(" ".join(X_test).split()) - train_words
saved_model = joblib.load(MODEL_PATH)
for label, vec in [("fresh", fresh_tfidf.named_steps["tfidf"]),
                   ("saved", saved_model.named_steps["tfidf"])]:
    vocab_words = {w for term in vec.vocabulary_ for w in term.split()}
    check(5, f"TF-IDF ({label}) vocabulary has no test-only words",
          len(vocab_words & test_only) == 0,
          f"{len(vocab_words)} vocabulary words, {len(test_only)} test-only words")

# ---- 6. Model trains successfully (fresh copy, saved file untouched) ----
fresh_model = build_model_pipeline()
fresh_model.fit(X_train, y_train)
check(6, "Model trains successfully",
      list(fresh_model.classes_) == ["Negative", "Positive"],
      f"classes: {list(fresh_model.classes_)}")

# ---- 7. Predictions work ----
y_pred = saved_model.predict(X_test)
check(7, "Predictions work",
      len(y_pred) == len(y_test) and set(y_pred) <= {"Positive", "Negative"},
      f"{len(y_pred)} predictions")

# ---- 8. Confusion matrix is generated ----
cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Positive"])
fig_path = "outputs/figures/confusion_matrix.png"
check(8, "Confusion matrix generated (counts and saved figure)",
      cm.sum() == len(y_test) and os.path.exists(fig_path)
      and os.path.getsize(fig_path) > 0,
      f"TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

# ---- 9. Precision, Recall, F1 are generated ----
p = precision_score(y_test, y_pred, pos_label="Positive")
r = recall_score(y_test, y_pred, pos_label="Positive")
f = f1_score(y_test, y_pred, pos_label="Positive")
check(9, "Precision, Recall, F1 computed from real predictions",
      all(0 <= v <= 1 for v in (p, r, f))
      and os.path.exists("outputs/reports/evaluation_report.txt"),
      f"Accuracy={accuracy_score(y_test, y_pred):.4f} Precision={p:.4f} "
      f"Recall={r:.4f} F1={f:.4f}")

# ---- 10. Saved model loads and matches a freshly trained one ----
fresh_pred = fresh_model.predict(X_test)
agreement = float((fresh_pred == y_pred).mean())
check(10, "Saved model loads and matches a fresh training run",
      agreement >= 0.999, f"agreement with fresh model: {agreement:.2%}")

# ---- 11. Streamlit application starts successfully ----
# (a) headless test of the app logic
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(APP_PATH, default_timeout=30).run()
at.text_area[0].input("Excellent product. The quality is amazing.").run()
at.button[0].click().run()
app_ok = (not at.exception) and len(at.success) == 1
check(11, "Streamlit app runs headlessly and shows a result", app_ok)

# (b) start the real server and ask its health endpoint
port = "8599"
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py",
     "--server.headless", "true", "--server.port", port],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
server_ok = False
try:
    for _ in range(30):
        time.sleep(1)
        try:
            reply = urllib.request.urlopen(
                f"http://localhost:{port}/_stcore/health", timeout=2).read()
            if reply == b"ok":
                server_ok = True
                break
        except Exception:
            continue
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
check(11, "Streamlit server starts and reports healthy", server_ok)

# ---- 12. New reviews classified without retraining ----
modified_before = os.path.getmtime(MODEL_PATH)
required = [
    ("Clearly positive", "Excellent product. The quality is amazing.", "Positive"),
    ("Clearly negative", "Terrible product. It stopped working after two days.", "Negative"),
    ("Negation", "The product is not good.", None),
    ("Mixed review", "The product quality is good but delivery was very late.", None),
    ("Short review", "Excellent!", None),
]
print("\n" + "-" * 60)
print("ACTUAL PREDICTIONS ON THE 5 REQUIRED REVIEWS")
print("-" * 60)
no_crash = True
clear_cases_ok = True
for kind, review, expected in required:
    try:
        sentiment, confidence = predict_sentiment(review)
    except Exception as error:
        no_crash = False
        print(f"{kind}: ERROR {error}")
        continue
    note = ""
    if expected is not None:
        ok = sentiment == expected
        clear_cases_ok &= ok
        note = f"  (expected {expected}: {'match' if ok else 'MISMATCH'})"
    print(f"[{kind}] {review}\n   -> {sentiment} ({confidence:.2%}){note}")
print("-" * 60)

unchanged = os.path.getmtime(MODEL_PATH) == modified_before
check(12, "New reviews classified without retraining",
      no_crash and unchanged and load_model() is load_model(),
      "model file unchanged, same cached model object reused")
check(12, "Clearly positive/negative reviews classified as expected",
      clear_cases_ok)

# ---- Summary ----
print("\n" + "=" * 60)
print(f"{sum(results)}/{len(results)} checks passed")
print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED - see [FAIL] lines above")
print("Negation, mixed and short reviews are reported, not asserted.")
sys.exit(0 if all(results) else 1)