# src/preprocessing.py
"""
Module for text preprocessing functions, primarily using spaCy.

This module handles loading the spaCy model and provides functions for
text cleaning, tokenization, lemmatization, and stopword/punctuation removal.
It also includes a utility for TF-IDF vectorization.
"""
from typing import List, Optional, Any, Iterable
import spacy
from spacy.language import Language
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# --- spaCy Model Loading ---
NLP_SPACY: Optional[Language] = None
try:
    NLP_SPACY = spacy.load('en_core_web_sm')
    print("spaCy 'en_core_web_sm' model loaded successfully.")
except OSError:
    print("CRITICAL ERROR: spaCy 'en_core_web_sm' model not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    # In a production system, you might raise an exception or exit here.
    # For this project, NLP_SPACY remains None, and functions should handle it.

# NLTK related code is commented out as spaCy is the primary tool now.
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize


def preprocess_text_spacy(text: str) -> str:
    """
    Preprocesses a single text string using spaCy.

    The preprocessing pipeline includes:
    1. Tokenization.
    2. Lemmatization of each token.
    3. Conversion to lowercase.
    4. Removal of stopwords.
    5. Removal of punctuation.
    6. Removal of space tokens.
    7. Removal of tokens that resemble numbers.

    Args:
        text (str): The input text string to preprocess.

    Returns:
        str: A string of space-separated processed (lemmatized, cleaned) tokens.
             Returns an empty string if input is empty or spaCy model is not loaded.
    """
    if not text:
        return ""
    if NLP_SPACY is None:
        print("spaCy model (NLP_SPACY) not loaded. Cannot preprocess text.")
        return ""

    doc = NLP_SPACY(text)

    processed_tokens: List[str] = [
        token.lemma_.lower().strip() for token in doc
        if not token.is_stop and \
           not token.is_punct and \
           not token.is_space and \
           not token.like_num  # Checks if the token consists of digits, e.g. "123"
    ]
    # Filter out any empty strings that might have resulted from stripping after lemmatization
    processed_tokens = [token for token in processed_tokens if token]

    return " ".join(processed_tokens)


def get_tfidf_vectorizer(
    texts: Optional[Iterable[str]] = None,
    max_features: int = 5000,
    **kwargs: Any
) -> TfidfVectorizer:
    """
    Creates and optionally fits a TF-IDF vectorizer.

    If 'texts' are provided, the vectorizer is fitted to these texts.
    Otherwise, an unfitted vectorizer is returned.

    Args:
        texts (Optional[Iterable[str]], optional): An iterable of text documents to fit the vectorizer on.
            Each element should be a preprocessed string of tokens. Defaults to None.
        max_features (int, optional): Maximum number of features (tokens) to keep, based on term frequency.
            Defaults to 5000.
        **kwargs (Any): Additional keyword arguments to pass to the TfidfVectorizer constructor.

    Returns:
        TfidfVectorizer: A scikit-learn TfidfVectorizer object (fitted or unfitted).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, **kwargs)
    if texts is not None:
        vectorizer.fit(texts)
    return vectorizer

# --- Main block for example usage and testing ---
if __name__ == '__main__':
    # This block demonstrates the preprocessing and TF-IDF vectorization.
    # It requires the spaCy model to be loaded.

    if NLP_SPACY is None:
        print("spaCy model not loaded. Skipping __main__ example in preprocessing.py.")
    else:
        sample_text: str = "This is a Sample Text for testing preprocessing! It includes numbers 123 and punctuation like $."

        print(f"Original Text:\n'{sample_text}'")

        processed_spacy_text: str = preprocess_text_spacy(sample_text)
        print(f"\nspaCy Preprocessed Text:\n'{processed_spacy_text}'")

        print("\n--- TF-IDF Vectorizer Example ---")
        corpus: List[str] = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
            "A very interesting document with interesting content."
        ]
        print(f"\nOriginal Corpus:\n{corpus}")

        preprocessed_corpus: List[str] = [preprocess_text_spacy(doc) for doc in corpus]
        print(f"\nPreprocessed Corpus for TF-IDF:\n{preprocessed_corpus}")

        # Fit a new TF-IDF vectorizer on the preprocessed corpus
        tfidf_vectorizer_example: TfidfVectorizer = get_tfidf_vectorizer(preprocessed_corpus, max_features=10)

        # Transform the corpus using the fitted vectorizer
        tfidf_matrix = tfidf_vectorizer_example.transform(preprocessed_corpus)

        print("\nTF-IDF Feature Names (Vocabulary):")
        try:
            # get_feature_names_out is preferred for scikit-learn >= 1.0
            print(tfidf_vectorizer_example.get_feature_names_out())
        except AttributeError:
            # Fallback for older scikit-learn versions
            print(tfidf_vectorizer_example.feature_names_) # type: ignore

        print("\nTF-IDF Matrix Shape:")
        print(tfidf_matrix.shape) # (number of documents, number of features)

        print("\nTF-IDF for the first document (sparse matrix representation):")
        print(tfidf_matrix[0])

        print("\nTF-IDF for the first document (dense array):")
        print(tfidf_matrix[0].toarray())
```
