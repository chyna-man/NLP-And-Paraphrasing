import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download("stopwords")
#nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Clean, tokenize, remove stopwords, lemmatize, and return cleaned text."""
    if not isinstance(text, str):
        return text
    
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))

    text = " ".join(text.split())

    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word not in stop_words]

    lemmatized_text = " ".join(lemmatizer.lemmatize(word) for word in filtered_words)
    
    return lemmatized_text

if __name__ == "__main__":
    df = pd.read_csv("synthetic_va_dataset.txt", sep='\t', quoting=3, encoding="utf-8")
    df['#1 String'] = df['#1 String'].apply(preprocess_text)
    df['#2 String'] = df['#2 String'].apply(preprocess_text)

df.to_csv("synthetic_va_dataset.txt_cleaned.txt", sep='\t', index=False)