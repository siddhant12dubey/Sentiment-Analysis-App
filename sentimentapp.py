# Sentiment Analysis of Social Media Posts Using AI/ML

import pandas as pd
import numpy as np
import re
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK stopwords (only once)
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[\d]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load dataset (Twitter Sentiment140 sample)
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv'
    df = pd.read_csv(url, encoding='latin-1')
    df = df.rename(columns={"label": "target", "tweet": "text"})
    df = df[['text', 'target']]
    df['target'] = df['target'].replace({4: 1})
    df['clean_text'] = df['text'].apply(clean_text)
    return df

# Train model
@st.cache_resource
def train_model(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['clean_text']).toarray()
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, vectorizer, acc

# Load and process data
data = load_data()
model, vectorizer, accuracy = train_model(data)

# Streamlit Web App
st.title("Sentiment Analysis of Social Media Posts")
st.write("Enter a sentence below to check its sentiment.")
st.write(f"Model Accuracy: **{accuracy:.2f}**")

user_input = st.text_input("Type your social media post here")
if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    st.write(f"Predicted Sentiment: **{sentiment}**")

# Optional to suppress deprecation warning
if __name__ == "__main__":
    pass  # Placeholder since there's no actual code here
   # st.set_option('deprecation.showfileUploaderEncoding', False)
