import pickle
import re

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved tokenizer
with open('models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the encoder
with open('models/encoder.npy', 'rb') as f:
    enc_categories = np.load(f, allow_pickle=True)

# Load the model
loaded_model = pickle.load(open('models/model.pkl', 'rb'))


# Function to preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.split()
    return text


# Get test data (replace with your actual test data)
test_data = [
    "Emmy Awards Viewership Dips To A Record-Low As Its Audience Continues To Drop",
    "Electability Is On The Ballot In Key Rhode Island House Primary",
]

# Preprocess the test data
processed_test_data = [preprocess_text(text) for text in test_data]
print(f"Processed Test Data: {processed_test_data}")
# Convert text data to numerical sequences
sequences = tokenizer.texts_to_sequences(processed_test_data)
features = pad_sequences(sequences)

# Make predictions using the loaded model
predictions = loaded_model.predict(features)
print(f"Predictions: {predictions}")
# Decode predictions back to category labels
print(f"Predicted categories: {np.argmax(predictions, axis=1)}")
predicted_categories = enc_categories[0][np.argmax(predictions, axis=1)]

# Print the predicted categories
print("Predicted categories:", predicted_categories)
