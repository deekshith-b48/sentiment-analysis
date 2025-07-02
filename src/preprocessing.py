# src/preprocessing.py
"""
Module for text preprocessing functions.
Includes functions for:
- Lowercasing
- Tokenization
- Stopword removal
- Lemmatization
- TF-IDF Vectorization
- (Later) Embeddings
"""

import nltk
# import spacy # Currently unused, comment out to avoid ModuleNotFoundError if not installed
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# NLTK resource download attempts commented out as we are moving to spaCy for core NLP tasks.
# Users should ensure NLTK resources are available if any minor NLTK utilities are used elsewhere.
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords', quiet=True)
# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet', quiet=True)
# try:
#     nltk.data.find('corpora/omw-1.4')
# except LookupError:
#     nltk.download('omw-1.4', quiet=True)
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt', quiet=True)
# try:
#     sent_detector = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
#     print("Successfully loaded 'tokenizers/punkt/PY3/english.pickle'. Punkt should be available.")
# except LookupError:
#     print("INFO: Failed to explicitly load 'tokenizers/punkt/PY3/english.pickle'.")
# except Exception as e:
#     print(f"WARNING: An unexpected error occurred while trying to load 'tokenizers/punkt/PY3/english.pickle': {e}")

# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize


# --- spaCy Preprocessing ---
import spacy

# Load spaCy model.
# This expects 'en_core_web_sm' to be downloaded: python -m spacy download en_core_web_sm
NLP_SPACY = None
try:
    NLP_SPACY = spacy.load('en_core_web_sm')
    print("spaCy 'en_core_web_sm' model loaded successfully.")
except OSError:
    print("CRITICAL ERROR: spaCy 'en_core_web_sm' model not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    # Exiting or raising an error might be appropriate here if spaCy is critical
    # For now, NLP_SPACY will remain None, and functions using it should handle this.

def preprocess_text_spacy(text: str) -> str:
    """
    Preprocesses text using spaCy:
    1. Lowercasing (handled by token attributes)
    2. Tokenization
    3. Lemmatization
    4. Stopword removal
    5. Punctuation removal
    6. Number removal
    """
    if not text or NLP_SPACY is None:
        if NLP_SPACY is None:
            print("spaCy model not loaded, cannot preprocess.")
        return ""

    doc = NLP_SPACY(text) # Process the text with spaCy

    processed_tokens = [
        token.lemma_.lower().strip() for token in doc
        if not token.is_stop and \
           not token.is_punct and \
           not token.is_space and \
           not token.like_num # Removes tokens that resemble numbers
    ]
    # Filter out any empty strings that might result from stripping
    processed_tokens = [token for token in processed_tokens if token]

    return " ".join(processed_tokens)

# Comment out NLTK based preprocessing
# def preprocess_text_nltk(text: str) -> str:
    """
    Preprocesses text using NLTK:
    1. Lowercasing
    2. Tokenization
    3. Stopword removal
    4. Lemmatization
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.strip()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return " ".join(lemmatized_tokens)

# def preprocess_text_spacy(text: str) -> str:
#     """
#     Preprocesses text using spaCy:
#     1. Lowercasing (spaCy handles this well)
#     2. Tokenization
#     3. Stopword removal
#     4. Lemmatization
#     """
#     if not text or not nlp_spacy:
#         return ""

#     doc = nlp_spacy(text.lower())

#     processed_tokens = [
#         token.lemma_ for token in doc
#         if not token.is_stop and not token.is_punct and not token.is_space
#     ]
#     return " ".join(processed_tokens)

def get_tfidf_vectorizer(texts=None, max_features=5000, **kwargs):
    """
    Creates and fits a TF-IDF vectorizer.
    If texts are provided, it fits the vectorizer to these texts.
    Otherwise, it returns an unfitted vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, **kwargs)
    if texts:
        vectorizer.fit(texts)
    return vectorizer

if __name__ == '__main__':
    sample_text = "This is a Sample Text for testing preprocessing! It includes numbers 123 and punctuation."

    print("Original Text:")
    print(sample_text)

    print("\nNLTK Preprocessed Text:")
    processed_nltk = preprocess_text_nltk(sample_text)
    print(processed_nltk)

    # print("\nspaCy Preprocessed Text:")
    # if nlp_spacy:
    #     processed_spacy = preprocess_text_spacy(sample_text)
    #     print(processed_spacy)
    # else:
    #     print("spaCy not available for testing.")

    # Example of TF-IDF
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    preprocessed_corpus = [preprocess_text_nltk(doc) for doc in corpus]
    print("\nPreprocessed Corpus for TF-IDF:")
    for doc in preprocessed_corpus:
        print(doc)

    tfidf_vectorizer = get_tfidf_vectorizer(preprocessed_corpus)
    tfidf_matrix = tfidf_vectorizer.transform(preprocessed_corpus)

    print("\nTF-IDF Feature Names (Vocabulary):")
    print(tfidf_vectorizer.get_feature_names_out()[:10]) # Print first 10 features

    print("\nTF-IDF Matrix Shape:")
    print(tfidf_matrix.shape)

    print("\nTF-IDF for the first document:")
    print(tfidf_matrix[0])
