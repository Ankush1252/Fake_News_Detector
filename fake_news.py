import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("./fake_news.csv")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25)

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training set
vectorizer.fit(X_train)

# Transform the training and testing sets
X_train_vectorized = vectorizer.transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create a LogisticRegression model
model = LogisticRegression()

# Fit the model to the training set
model.fit(X_train_vectorized, y_train)

# Predict the labels for the testing set
y_pred = model.predict(X_test_vectorized)

# Calculate the accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.3f}")

# Create a Streamlit app
st.title("Fake News Detector")

# Add a text input field to the app
text = st.text_input("Enter text here:")

# Add a button to the app that will predict the class of the text
if st.button("Detect"):
    # Vectorize the text
    text_vectorized = vectorizer.transform([text])

    # Predict the class of the text
    prediction = model.predict(text_vectorized)

    # Display the prediction to the user
    st.write("The news is predicted to be:", prediction[0])

