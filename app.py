import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load trained CNN model
model = load_model("cnn_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Constants
MAX_SEQUENCE_LENGTH = 100

# Streamlit UI
st.title("Text Classification using CNN")
st.write("Enter a text below and let the model classify it.")

user_input = st.text_area("Enter your text here:")

if st.button("Classify"):
    if user_input:
        # Preprocess input text
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded_sequence)
        predicted_class = np.argmax(prediction)
        
        # Map prediction to labels
        label = "Fake News" if predicted_class == 0 else "Real News"
        
        st.success(f"Predicted Class: {label}")
    else:
        st.warning("Please enter some text to classify.")
