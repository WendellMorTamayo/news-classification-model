import pickle
import tensorflow as tf
import streamlit as st
import numpy as np
import nltk
import tensorflow
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# model = tf.keras.models.load_model('models/mod.keras')

# Load the saved tokenizer
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
    # text = param.lower()
    # text = re.sub(r"[^\w\s]", '', text)
    # text = nltk.word_tokenize(text)
    #
    # stop_words = set(stopwords.words("english"))
    # text = [word for word in param if word not in stop_words]
    #
    # lemmatizer = WordNetLemmatizer()
    #
    # text = " ".join([lemmatizer.lemmatize(word) for word in text])
    #
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(text)
    # sequences = tokenizer.texts_to_sequences([text])
    # features = pad_sequences(sequences)
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
# import re
#
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.saved_model import load
# import pickle
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder
#
# # Load the tokenizer
# with open('models/tokenizer.pkl', 'rb') as file:
#     tokenizer = pickle.load(file)
#
# # Load the model
# # model = load('models/model')
# model = tf.keras.models.load_model('models/mod.keras')
#
# # Load the encoder
# encoder = OneHotEncoder(handle_unknown="ignore")
# with open('models/encoder.npy', 'rb') as f:
#     encoder.categories_ = np.load(f, allow_pickle=True)
#
#
# def predict_category(text):
#     # Preprocess the text
#     text = text.lower()
#     text = re.sub(r"[^\w\s]", "", text)
#     text = tokenizer.texts_to_sequences([text])
#     text = pad_sequences(text)
#
#     # Predict the category
#     pred = model.predict(text)
#     pred = np.argmax(pred)
#     category = encoder.inverse_transform([[pred]])[0][0]
#     return category
#
#
# st.title("News Category Prediction App")
# st.write("This app predicts the category of a news article based on its headline and short description.")
#
# text_input = st.text_area("Enter Headline and Short Description:", height=100)
#
# if st.button("Predict Category"):
#     category = predict_category(text_input)
#     st.write("Predicted Category:", category)
