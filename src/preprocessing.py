import pandas as pd
import numpy as np
import re
import string

from collections import Counter

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
def clean_text(text):

    # -----------------------------------------------------
    # Convert to lowercase
    # -----------------------------------------------------

    text = text.lower()

    # -----------------------------------------------------
    # Remove URLs
    # -----------------------------------------------------

    text = re.sub(r'http\S+|www\S+', '', text)

    # -----------------------------------------------------
    # Remove HTML tags
    # -----------------------------------------------------

    text = re.sub(r'<.*?>', '', text)

    # -----------------------------------------------------
    # Remove punctuation
    # -----------------------------------------------------

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # -----------------------------------------------------
    # Remove numbers
    # -----------------------------------------------------

    text = re.sub(r'\d+', '', text)

    # -----------------------------------------------------
    # Remove extra whitespace
    # -----------------------------------------------------

    text = re.sub(r'\s+', ' ', text).strip()

    return text
# =========================================================
# 9. TOKENIZATION + STOPWORD REMOVAL + LEMMATIZATION
# =========================================================

def preprocess_text(text):

    # Clean basic text
    text = clean_text(text)

    # -----------------------------------------------------
    # Tokenization using regex
    # -----------------------------------------------------

    tokens = re.findall(r'\w+', text)

    # -----------------------------------------------------
    # Stopword Removal
    # -----------------------------------------------------

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # -----------------------------------------------------
    # Lemmatization
    # -----------------------------------------------------

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # -----------------------------------------------------
    # Join back to sentence
    # -----------------------------------------------------

    processed_text = " ".join(tokens)

    return processed_text
