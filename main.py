# ================================
# Import Libraries
# ================================
import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ================================
# Load IMDB Dataset
# ================================
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# ================================
# Load Trained Model
# ================================
# Recommended:
# model = load_model("simple_rnn_imdb.keras")

# If you have only .h5
model = load_model("simple_rnn.h5")


# ================================
# Helper Functions
# ================================
def decode_review(encoded_review):
    return " ".join(
        [reverse_word_index.get(i - 3, "?") for i in encoded_review]
    )


def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=500,
        padding="pre",
        truncating="pre"
    )

    return padded_review


# ================================
# Streamlit UI
# ================================
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬"
)

st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the model will predict whether it is Positive or Negative."
)

user_input = st.text_area(
    "Movie Review",
    height=200
)

if st.button("Classify"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:

        review = preprocess_text(user_input)

        prediction = model.predict(review, verbose=0)

        score = float(prediction[0][0])

        sentiment = "😊 Positive" if score >= 0.5 else "😞 Negative"

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: **{score:.4f}**")