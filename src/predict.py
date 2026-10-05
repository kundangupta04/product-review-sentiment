
import joblib

from src.preprocessing import clean_text

MODEL_PATH = "models/sentiment_pipeline.joblib"

_model = None  # cache: the model is loaded from disk only once


def load_model(path=MODEL_PATH):
    """Load the saved pipeline the first time, then reuse it."""
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def predict_sentiment(review_text):
    """Take a raw review and return (sentiment, confidence).

    sentiment  : "Positive" or "Negative"
    confidence : probability of the predicted class (0.0 to 1.0)
    """
    cleaned = clean_text(review_text)           # same cleaning as training
    if cleaned == "":
        raise ValueError("Review is empty after cleaning. Please enter some text.")

    model = load_model()
    probabilities = model.predict_proba([cleaned])[0]   # one probability per class
    best = probabilities.argmax()
    sentiment = model.classes_[best]
    confidence = float(probabilities[best])
    return sentiment, confidence


if __name__ == "__main__":
    print(predict_sentiment("The product quality is excellent and I am very happy with it."))