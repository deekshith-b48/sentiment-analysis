# src/interpretability.py
"""
Module for model interpretability using LIME (Local Interpretable Model-agnostic Explanations).
"""
from typing import Callable, List, Tuple, Optional, Any, Sequence

import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
# SklearnModel type can be imported or defined if a common type hint is used across modules
# from .models import SklearnModel # Example if SklearnModel is defined in models.py
SklearnModel = Any # Using Any for simplicity here, replace with more specific type if available


# Type alias for the preprocessor function
PreprocessorFunc = Callable[[str], str]
# Type alias for the LIME predictor function
LimePredictorFunc = Callable[[List[str]], np.ndarray]


def explain_instance_lime(
    raw_text: str,
    model: SklearnModel,
    vectorizer: TfidfVectorizer,
    preprocessor_func: PreprocessorFunc,
    class_names: Optional[List[str]] = None,
    num_features: int = 10
) -> Optional[List[Tuple[str, float]]]:
    """
    Explains a single prediction for a given raw text instance using LIME.

    Args:
        raw_text (str): The raw text string to explain.
        model (SklearnModel): The trained scikit-learn compatible classifier.
            Must have a `predict_proba` method.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        preprocessor_func (PreprocessorFunc): A function that takes a raw text string
            and returns a preprocessed string of tokens.
        class_names (Optional[List[str]], optional): List of class names that correspond
            to the order of probabilities returned by model.predict_proba().
            If None, LIME uses generic names like "Class 0", "Class 1".
            For binary sentiment, this is typically ['negative', 'positive'].
            Defaults to None.
        num_features (int, optional): The number of top features (words) to include
            in the explanation. Defaults to 10.

    Returns:
        Optional[List[Tuple[str, float]]]: A list of (feature, weight) tuples representing
            the LIME explanation for the top predicted class. Each tuple contains a
            word (feature) and its corresponding weight indicating its contribution.
            Returns None if an error occurs during explanation.
    """
    if not raw_text:
        print("Error: Raw text for LIME explanation cannot be empty.")
        return None

    explainer = LimeTextExplainer(class_names=class_names)

    def predictor(texts: List[str]) -> np.ndarray:
        """
        LIME-compatible predictor function.
        Takes a list of raw text strings, preprocesses and vectorizes them,
        and returns prediction probabilities from the model.
        """
        if not texts:
            return np.array([])

        processed_texts: List[str] = [preprocessor_func(text) for text in texts]

        try:
            vectorized_texts = vectorizer.transform(processed_texts)
        except Exception as e:
            print(f"Error during vectorization in LIME predictor: {e}")
            num_classes_fallback = len(class_names) if class_names else (model.classes_.shape[0] if hasattr(model, 'classes_') else 2)
            return np.full((len(texts), num_classes_fallback), 1.0 / num_classes_fallback)

        if hasattr(model, 'predict_proba'):
            try:
                probas: np.ndarray = model.predict_proba(vectorized_texts)
                return probas
            except Exception as e:
                print(f"Error during model.predict_proba in LIME predictor: {e}")
                num_classes_fallback = len(class_names) if class_names else (model.classes_.shape[0] if hasattr(model, 'classes_') else 2)
                return np.full((len(texts), num_classes_fallback), 1.0 / num_classes_fallback)
        else:
            print("CRITICAL: Model does not have predict_proba method. LIME requires probability scores.")
            # This case should ideally not be reached if model compatibility is ensured.
            num_classes_fallback = len(class_names) if class_names else (model.classes_.shape[0] if hasattr(model, 'classes_') else 2)
            return np.full((len(texts), num_classes_fallback), 1.0 / num_classes_fallback)

    try:
        explanation = explainer.explain_instance(
            text_instance=raw_text,
            classifier_fn=predictor,
            num_features=num_features,
            top_labels=1 # Explain only the top predicted label
        )

        # explanation.as_list() returns features for the label passed, or top label if None
        # explanation.top_labels[0] gives the index of the top predicted class by LIME's internal call
        predicted_class_idx: int = explanation.top_labels[0]
        explanation_list: List[Tuple[str, float]] = explanation.as_list(label=predicted_class_idx)

        return explanation_list

    except Exception as e:
        print(f"Error during LIME explanation generation for text '{raw_text[:50]}...': {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # This block demonstrates LIME explanation.
    # It requires a spaCy model for preprocessing and scikit-learn for model/vectorizer.

    from sklearn.linear_model import LogisticRegression
    import sys, os

    # Ensure src modules can be imported when running this file directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.dirname(current_dir) # up to 'src'
    if src_path not in sys.path:
        sys.path.append(src_path)

    try:
        from preprocessing import preprocess_text_spacy, NLP_SPACY
    except ImportError:
        print("Could not import preprocessing module. Ensure it's in the Python path.")
        NLP_SPACY = None # Ensure NLP_SPACY is defined for the check below

    if NLP_SPACY is None:
        print("spaCy model 'en_core_web_sm' (NLP_SPACY) is not loaded from preprocessing.py.")
        print("Cannot run LIME __main__ example. Ensure spaCy model is downloaded and loads correctly.")
    else:
        print("Setting up mock objects for LIME explainer __main__ test...")

        sample_corpus_main: List[str] = [
            "This is a wonderfully fantastic movie, I loved it!",
            "This film was terribly boring and a complete waste of time.",
            "The movie was okay, neither good nor bad."
        ]
        # Corresponding labels (1 for positive, 0 for negative)
        sample_labels_main: List[int] = [1, 0, 1]

        print("Preprocessing sample data for __main__ test...")
        processed_corpus_main: List[str] = [preprocess_text_spacy(text) for text in sample_corpus_main]

        if not all(s.strip() for s in processed_corpus_main if s is not None): # Check if all are non-empty strings
            print("Mock LIME test in __main__: Preprocessing failed for some texts or produced empty strings.")
        else:
            mock_vectorizer_main = TfidfVectorizer()
            mock_vectorizer_main.fit(processed_corpus_main)
            X_mock_main = mock_vectorizer_main.transform(processed_corpus_main)

            mock_model_main = LogisticRegression()
            mock_model_main.fit(X_mock_main, sample_labels_main)
            print("Mock model trained for __main__ test.")

            test_raw_text_positive_main: str = "This is an amazing and fantastic piece of art, truly enjoyable."
            test_raw_text_negative_main: str = "I absolutely hated this, it's a boring and terrible waste of my time."

            # Class names should match the model's output interpretation (0: negative, 1: positive)
            class_names_main: List[str] = ['negative', 'positive']

            print(f"\nExplaining positive text for __main__: '{test_raw_text_positive_main}'")
            explanation_pos_main = explain_instance_lime(
                raw_text=test_raw_text_positive_main,
                model=mock_model_main,
                vectorizer=mock_vectorizer_main,
                preprocessor_func=preprocess_text_spacy,
                class_names=class_names_main,
                num_features=5
            )
            if explanation_pos_main:
                print("LIME Explanation (Positive - __main__):")
                for feature, weight in explanation_pos_main:
                    print(f"  '{feature}': {weight:.4f}")
            else:
                print("Failed to get LIME explanation for positive text in __main__.")

            print(f"\nExplaining negative text for __main__: '{test_raw_text_negative_main}'")
            explanation_neg_main = explain_instance_lime(
                raw_text=test_raw_text_negative_main,
                model=mock_model_main,
                vectorizer=mock_vectorizer_main,
                preprocessor_func=preprocess_text_spacy,
                class_names=class_names_main,
                num_features=5
            )
            if explanation_neg_main:
                print("LIME Explanation (Negative - __main__):")
                for feature, weight in explanation_neg_main:
                    print(f"  '{feature}': {weight:.4f}")
            else:
                print("Failed to get LIME explanation for negative text in __main__.")
```
