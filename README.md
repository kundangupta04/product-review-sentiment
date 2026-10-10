# Product Review Sentiment Analysis using TF-IDF and Machine Learning

A beginner-friendly NLP project that predicts whether a product review is
**Positive** or **Negative**. It uses TF-IDF features and a Logistic Regression
classifier, with a simple Streamlit web demo.

---


## Problem Statement
Online stores receive thousands of product reviews. Reading them all by hand is
slow and impractical. This project builds a system that reads a review and
automatically labels its sentiment as Positive or Negative.

## Objective
Build a classical machine learning pipeline that:
- cleans raw review text,
- converts it to numbers using TF-IDF,
- trains a Logistic Regression classifier,
- evaluates it with Accuracy, Precision, Recall, F1-score and a confusion matrix,
- predicts the sentiment of a new, user-entered review through a simple web interface.

## Features
- Text cleaning (HTML, URLs, punctuation, case) that **keeps negation words**
- Handling of missing values and duplicate reviews
- Exploratory data analysis with saved charts
- Stratified 80/20 train/test split with a fixed `random_state`
- TF-IDF features inside a scikit-learn `Pipeline` (no data leakage)
- Logistic Regression classifier
- Evaluation: Accuracy, Precision, Recall, F1-score, confusion matrix, classification report
- Reusable prediction function that returns the sentiment and a confidence score
- Streamlit demo (text box, button, result, confidence)
- Optional comparison with Multinomial Naive Bayes
- Automated end-to-end test script

## Technology Stack
| Tool | Used for |
|---|---|
| Python | Main language |
| Pandas, NumPy | Data handling |
| Scikit-learn | TF-IDF, Logistic Regression, Naive Bayes, metrics, Pipeline |
| Matplotlib, Seaborn | Charts and confusion matrix |
| WordCloud | Optional word cloud charts |
| Streamlit | Demo interface |
| Joblib | Saving and loading the trained pipeline |

NLTK is listed in `requirements.txt` because the project brief allows it, but
the current code does **not** use it. No deep learning libraries are used.

## Dataset
- **Source:** Amazon Fine Food Reviews (Kaggle): https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- **File used:** `Reviews.csv`
- **Columns used:** `Text` (the review) and `Score` (1 to 5 stars)
- **Labels:**
  - 4 to 5 stars = **Positive**
  - 1 to 2 stars = **Negative**
  - 3-star (neutral) reviews are removed, so the task is binary
- **Sample:** 20,000 reviews, selected with `random_state=42`
- **After cleaning:** 19,289 reviews remain (711 duplicate reviews removed)
  - Positive: 16,309 (84.55%)
  - Negative: 2,980 (15.45%)
- **Split:** 15,431 training reviews and 3,858 test reviews

The dataset is **imbalanced**, with far more Positive than Negative reviews.
The raw file is not included in this repository. Download it and place it in
`data/raw/`.


##  Project Structure
```
product-review-sentiment/
├── data/
│   ├── raw/                  # Reviews.csv (you download this)
│   └── processed/            # reviews_labeled.csv, reviews_cleaned.csv
├── notebooks/
├── models/                   # sentiment_pipeline.joblib
├── outputs/
│   ├── figures/              # EDA charts and confusion_matrix.png
│   └── reports/              # evaluation_report.txt, model_comparison.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # load raw data, create labels, inspect
│   ├── preprocessing.py      # clean_text() function
│   ├── clean_dataset.py      # apply cleaning, drop missing/empty/duplicates
│   ├── eda.py                # exploratory data analysis and charts
│   ├── features.py           # train/test split and TF-IDF pipeline
│   ├── train.py              # train Logistic Regression and save pipeline
│   ├── evaluate.py           # metrics and confusion matrix
│   ├── predict.py            # predict_sentiment() function
│   ├── compare_models.py     # optional Naive Bayes comparison
│   ├── test_preprocessing.py # tests for clean_text()
│   ├── test_predict.py       # tests for predict_sentiment()
│   ├── test_app.py           # headless test of the Streamlit app
│   └── test_project.py       # complete end-to-end test
├── app.py                    # Streamlit demo
├── check_setup.py            # checks libraries and folders
├── requirements.txt
├── README.md
└── .gitignore
```

## Data Preprocessing
The function `clean_text()` in `src/preprocessing.py` does the following:

1. Converts missing or non-text values to an empty string
2. Converts text to lowercase
3. Removes HTML tags (for example `<br />`)
4. Removes URLs
5. Expands contractions so negation is not lost (`don't` becomes `do not`,
   `can't` becomes `can not`, `won't` becomes `will not`)
6. Removes punctuation, digits and symbols
7. Collapses extra spaces

The words **not, no, never** are deliberately kept. "Not good" has the opposite
meaning of "good", so removing "not" would hurt sentiment classification.

When the dataset is cleaned (`src/clean_dataset.py`), rows with missing reviews,
reviews that are empty after cleaning, and duplicate reviews are dropped.
Duplicates are removed so the same review cannot appear in both the training
and test sets.

## TF-IDF
TF-IDF (Term Frequency, Inverse Document Frequency) turns text into numbers.
A word gets a **high** weight if it appears often in one review but rarely
across all reviews. A word gets a **low** weight if it is common everywhere.

Settings used (in `src/features.py`):
- `max_features=5000`: keep the 5,000 most useful terms
- `ngram_range=(1, 2)`: use single words and two-word phrases such as "not good"
- A **custom stopword list**: scikit-learn's built-in English stopword list
  contains "not", "no" and similar words. These were removed from the list so
  that negation information is kept.

**No data leakage:** the data is split first. The TF-IDF vectorizer is then
fitted only on the training set (`fit_transform`) and only applied to the test
set (`transform`). The project checks this explicitly: no word that appears
only in the test data ends up in the vocabulary.

## Logistic Regression
Logistic Regression gives each TF-IDF feature a weight, adds the weighted
values together, and converts the total into a probability between 0 and 1.
If the probability of Positive is above 0.5, the review is labeled Positive;
otherwise it is labeled Negative.

Settings used (in `src/train.py`):
- `max_iter=1000`: enough training iterations to converge
- `random_state=42`: reproducible results
- `class_weight="balanced"`: mistakes on the smaller Negative class count more
  during training, which helps with the class imbalance. It uses only the
  training labels.

## Evaluation Metrics
- **True Positive (TP):** actually Positive, predicted Positive
- **True Negative (TN):** actually Negative, predicted Negative
- **False Positive (FP):** actually Negative, predicted Positive
- **False Negative (FN):** actually Positive, predicted Negative

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP + TN) / total | Share of all predictions that were correct |
| Precision | TP / (TP + FP) | Of the reviews predicted as a class, how many really were |
| Recall | TP / (TP + FN) | Of the reviews that truly belong to a class, how many were found |
| F1-score | 2 x (Precision x Recall) / (Precision + Recall) | Balance of Precision and Recall |

**Results on the held-out test set (3,858 reviews), Logistic Regression:**

| Metric | Positive class | Negative class |
|---|---|---|
| Precision | 0.9695 | 0.6477 |
| Recall | 0.9163 | 0.8423 |
| F1-score | 0.9422 | 0.7323 |

- Overall **Accuracy: 0.9049**
- Macro F1: 0.8372
- A dummy model that always predicts "Positive" would score 0.8455 accuracy, so
  accuracy alone is misleading on this imbalanced data. The Negative-class
  numbers give a fuller picture.

All values come from real predictions on the test set. They are saved in
`outputs/reports/evaluation_report.txt`.

## Confusion Matrix
Rows are the actual label and columns are the predicted label:

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 502 (TN) | 94 (FP) |
| **Actual Positive** | 273 (FN) | 2989 (TP) |

The figure is saved at `outputs/figures/confusion_matrix.png`. Here "Positive"
is treated as the positive class. The model catches 84% of truly negative
reviews, but only 65% of the reviews it calls Negative really are Negative.
It wrongly labels 273 Positive reviews as Negative, which is the trade-off of
using `class_weight="balanced"`.

## Model Comparison
Multinomial Naive Bayes was trained on the same split and the same TF-IDF
settings, and measured on the same test set. Run it with
`python -m src.compare_models`.

| Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---|---|
| Accuracy | 0.9049 | 0.8730 |
| Precision (Positive) | 0.9695 | 0.8716 |
| Recall (Positive) | 0.9163 | 0.9966 |
| F1 (Positive) | 0.9422 | 0.9299 |
| Precision (Negative) | 0.6477 | 0.9141 |
| Recall (Negative) | 0.8423 | 0.1963 |
| F1 (Negative) | 0.7323 | 0.3232 |
| Macro F1 | 0.8372 | 0.6266 |

Confusion counts (TN / FP / FN / TP): Logistic Regression 502 / 94 / 273 / 2989,
Naive Bayes 117 / 479 / 11 / 3251.

**Reading the results:** On this test set, Logistic Regression scored higher on
Accuracy, Negative-class Recall and Macro F1. Naive Bayes scored higher on
Negative-class Precision and Positive-class Recall, but it labeled most reviews
Positive and found only about 20% of the truly negative reviews. Part of this
difference comes from class imbalance: Logistic Regression uses
`class_weight="balanced"` and Naive Bayes has no such setting, so the
comparison is not perfectly like-for-like. These results describe one split
and should not be read as a general ranking of the algorithms.

Logistic Regression remains the primary model of this project.

## Installation
Requires Python 3.10 or newer.

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```



Then download `Reviews.csv` from Kaggle (see Section 5) and place it in
`data/raw/`. Make sure `(venv)` appears at the start of your terminal line
before running any command.

## 15. How to Train the Model
Run these commands from the project root, in this order:

```bash
python src/data_loader.py       # load raw data, create labels
python -m src.clean_dataset     # clean text, drop missing/empty/duplicates
python -m src.eda               # optional: EDA charts
python -m src.features          # optional: check split and TF-IDF
python -m src.train             # train and save the model
python -m src.evaluate          # metrics and confusion matrix
```

The trained pipeline is saved to `models/sentiment_pipeline.joblib`.

## Run the Streamlit Application
```bash
python -m streamlit run app.py
```
Then open http://localhost:8501 in your browser, type a review and click
**Analyze Sentiment**. The model must be trained first (Section 15). The app
loads the saved pipeline once and never retrains.

## Input and Output
Real outputs from `python -m src.test_project`:

| Review | Prediction | Confidence |
|---|---|---|
| Excellent product. The quality is amazing. | Positive | 97.62% |
| Terrible product. It stopped working after two days. | Negative | 92.25% |
| The product is not good. | Negative | 95.98% |
| The product quality is good but delivery was very late. | Positive | 84.33% |
| Excellent! | Positive | 99.35% |

The mixed review is genuinely ambiguous. The model has only two classes, so it
had to pick one, and it leaned Positive with lower confidence than the clear
cases.

Using the prediction function directly:
```python
from src.predict import predict_sentiment

sentiment, confidence = predict_sentiment("The product is excellent and worth the money.")
print(sentiment, f"{confidence:.2%}")
```

## Testing
| Command | What it checks |
|---|---|
| `python check_setup.py` | Libraries import correctly and folders exist |
| `python -m src.test_preprocessing` | `clean_text()` on sample reviews, negation kept |
| `python -m src.test_predict` | Prediction function on several review types |
| `python -m src.test_app` | Streamlit app runs headlessly |
| `python -m src.test_project` | Full end-to-end test of all 12 project requirements |

The end-to-end test passed all 15 checks, including the data split, the TF-IDF
leakage check, model training, metrics, saved-model loading, the Streamlit app
and classification without retraining.

## Project Limitations
- The **Negative class is the weak spot** (Precision 0.6477), mainly caused by
  class imbalance.
- The model has only two classes, so **mixed reviews** are forced into one of
  them.
- Labels come from **star ratings**, not human judgment, so some labels are
  noisy.
- The data is **food reviews only**, so accuracy may drop on other product types.
- Results come from a **single 80/20 split** and a 20,000-review sample.
- TF-IDF ignores word order beyond two-word phrases and cannot understand
  sarcasm.
- Confidence is a model probability, not a guarantee of correctness.
- Neutral (3-star) reviews were removed, so the model never learned them.

## Future Scope
- Use the full dataset instead of a 20,000-review sample
- Try cross-validation and hyperparameter tuning
- Add a Neutral class
- Test on other product categories
- Compare with additional classical models
- Explore other methods only if the course allows them
