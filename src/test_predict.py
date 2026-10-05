import time

from src.predict import load_model, predict_sentiment

test_reviews = [
    ("Clearly positive", "The product quality is excellent and I am very happy with it."),
    ("Clearly negative", "The product stopped working after two days. Very disappointed."),
    ("Short review", "Excellent!"),
    ("Longer review",
     "I bought this a month ago and I have used it almost every day since. "
     "The quality is great, it is easy to use, and the price was fair. "
     "I would happily recommend it to my friends and family."),
    ("Negation", "The product is not good."),
    ("Negation 2", "I would never buy this again."),
]

print("=" * 60)
for kind, review in test_reviews:
    sentiment, confidence = predict_sentiment(review)
    print(f"[{kind}]")
    print(f"Review    : {review}")
    print(f"Sentiment : {sentiment}")
    print(f"Confidence: {confidence:.2%}\n")

# Edge case: empty input must give a clear error, not a crash
print("=" * 60)
for bad in ["", "!!!"]:
    try:
        predict_sentiment(bad)
    except ValueError as error:
        print(f"Empty input {bad!r} handled -> {error}")

# Proof that the model is cached (loaded once, not retrained)
print("=" * 60)
print("Same model object reused:", load_model() is load_model())
start = time.time()
for _ in range(100):
    predict_sentiment("Great value for the money.")
print(f"100 predictions took {time.time() - start:.2f} seconds (no retraining)")