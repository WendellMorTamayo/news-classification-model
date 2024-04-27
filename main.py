import matplotlib.pyplot as plt
import pickle

import numpy as np
import pandas as pd
import tensorflow
from sklearn.preprocessing import OneHotEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers

import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

filename = "news.json"
df = pd.read_json(filename, lines=True)
df = df.dropna()
df = df.drop(columns=["link"])
df = df.drop(columns=["date"])
df = df.drop(columns=["authors"])
categories = ["WELLNESS", "ENTERTAINMENT", "POLITICS", "TRAVEL", "STYLE & BEAUTY"]
df = df[df["category"].isin(categories)]

majority = df[df["category"] == "POLITICS"]
minorities = df[df["category"].isin(["TRAVEL", "STYLE & BEAUTY", "ENTERTAINMENT", "WELLNESS"])]

minorities_upsampled = [resample(df[df["category"] == cls],
                                 replace=True, n_samples=len(majority),
                                 random_state=123) for cls in minorities["category"].unique()]

df = pd.concat([majority] + minorities_upsampled)

sns.countplot(x="category",
              data=df,
              order=df.category.value_counts().index
              )

df["headline"] = df["headline"].str.lower()
df["category"] = df["category"].str.lower()
df["short_description"] = df["short_description"].str.lower()

df['headline'] = df['headline'].apply(lambda x: re.sub(r"[^\w\s]", "", x))
df['short_description'] = df['short_description'].apply(lambda x: re.sub(r"[^\w\s]", "", x))

df["headline"] = df["headline"].apply(nltk.word_tokenize)
df["short_description"] = df["short_description"].apply(nltk.word_tokenize)

stop_words = set(stopwords.words("english"))

df["headline"] = df["headline"].apply(lambda x: [word for word in x if word not in stop_words])
df["short_description"] = df["short_description"].apply(lambda x: [word for word in x if word not in stop_words])

lemmatizer = WordNetLemmatizer()

df["headline"] = df["headline"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x]))
df["short_description"] = df["short_description"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x]))

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df["headline"] + " " + df["short_description"])


with open('models/tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)


sequences = tokenizer.texts_to_sequences(df["headline"] + " " + df["short_description"])

features = pad_sequences(sequences)

enc = OneHotEncoder(handle_unknown="ignore")
print(enc.inverse_transform(enc.fit_transform(df[["category"]]).toarray()))
category = enc.fit_transform(df[["category"]]).toarray()
print(enc)
with open('models/encoder.npy', 'wb') as f:
    np.save(f, enc.categories_, allow_pickle=True)


x_train, x_test, y_train, y_test = train_test_split(features, category, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1

output_size = category.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, 64))
model.add(Bidirectional(LSTM(units=64)))
model.add(Dropout(rate=0.5))
model.add(Dense(units=64, activation="relu", kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(output_size, activation="softmax", kernel_regularizer=regularizers.l2(0.01)))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=False)

plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.legend()

model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)


def plot_cm(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.diverging_palette(220, 20, n=7),
        ax=ax

    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.show()


plot_cm(
    enc.inverse_transform(y_test),
    enc.inverse_transform(y_pred),
    enc.categories_[0]
)

# model.save('models/mod.keras')

# tensorflow.saved_model.save(model, "models/model.keras")
# tensorflow.saved_model.save(model, "models/model")
with open('models/models.pkl', 'wb') as file:
    pickle.dump(model, file)

