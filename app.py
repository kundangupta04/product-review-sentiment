"""Step 9: simple Streamlit demo for Product Review Sentiment Analysis."""
import os
import sys

# Make "from src..." work no matter how the app is started
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from src.predict import load_model, predict_sentiment

st.set_page_config(page_title="Product Review Sentiment Analysis", page_icon="🛒")


@st.cache_resource  # load the trained model once, not on every click
def get_model():
    return load_model()


st.title("Product Review Sentiment Analysis")
st.write(
    "Type a product review below. The app cleans the text, converts it to "
    "TF-IDF features, and a trained Logistic Regression model predicts "
    "whether the review is **Positive** or **Negative**."
)

# Load the model once; stop with a clear message if it has not been trained yet
try:
    get_model()
except FileNotFoundError:
    st.error("Trained model not found. Run `python -m src.train` first.")
    st.stop()

review = st.text_area(
    "Product Review",
    height=150,
    placeholder="Example: The product is excellent and worth the money.",
)

if st.button("Analyze Sentiment", type="primary"):
    try:
        sentiment, confidence = predict_sentiment(review)
    except ValueError as error:
        st.warning(str(error))
    else:
        st.subheader("Result")
        if sentiment == "Positive":
            st.success("POSITIVE")
        else:
            st.error("NEGATIVE")

        st.write(f"**Confidence:** {confidence:.2%}")
        st.progress(confidence)

        if confidence < 0.60:
            st.info("Low confidence. The review may be mixed or unclear.")

st.divider()
st.caption(
    "Model: TF-IDF + Logistic Regression, trained on Amazon Fine Food reviews. "
    "Predictions may be less reliable for other product types or mixed reviews."
)