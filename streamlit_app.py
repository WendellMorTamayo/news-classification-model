import pickle
import streamlit as st
import numpy as np
import re

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences


with open('models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the encoder
with open('models/encoder.npy', 'rb') as f:
    enc_categories = np.load(f, allow_pickle=True)

# Load the model
model = pickle.load(open('models/model.pkl', 'rb'))

encoder = OneHotEncoder(handle_unknown="ignore")
with open('models/encoder.npy', 'rb') as f:
    encoder.categories_ = np.load(f, allow_pickle=True)


def preprocess_input(param):
    text = param.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.split()
    # tokenizer = Tokenizer()
    sequences = tokenizer.texts_to_sequences([text])
    print(f"Tokenizer: {sequences}")
    features = pad_sequences(sequences)
    return features


def predict_category(param):
    processed_input = preprocess_input(param)
    print(f"Processed Input: {processed_input}")
    prediction = model.predict(processed_input)
    print(f"Prediction: {np.argmax(prediction, axis=1)}")
    print(f"Prediction Features: {encoder.categories_[0]}")

    return (encoder.categories_[0][np.argmax(prediction, axis=1)][0]).upper()


def main():
    st.title('News Category Predictor')

    user_input = st.text_area("Enter a news headline:", height=100)

    if st.button('Predict'):
        prediction = predict_category(user_input)
        st.write(f'The predicted category is: {prediction}')

    st.write("This app predicts the category of a news article based on its headline.")


if __name__ == "__main__":
    main()

