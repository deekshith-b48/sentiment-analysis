# src/interpretability.py
"""
Module for model interpretability using LIME.
"""
from lime.lime_text import LimeTextExplainer
import numpy as np

# Assuming preprocessing functions might be needed here if not handled before calling
# from .preprocessing import preprocess_text_spacy # Or however it's accessed

def explain_instance_lime(raw_text: str, model, vectorizer, preprocessor_func, class_names=None, num_features=10):
    """
    Explains a single prediction using LIME.

    Args:
        raw_text (str): The raw text instance to explain.
        model: The trained scikit-learn model.
        vectorizer: The fitted TF-IDF vectorizer.
        preprocessor_func: The function used to preprocess text (e.g., preprocess_text_spacy).
                           It should take raw text and return a string of processed tokens.
        class_names (list, optional): List of class names. If None, LIME will use generic names.
                                      For binary classification, usually ['negative', 'positive'].
        num_features (int, optional): Number of top features to show in the explanation.

    Returns:
        list: A list of (feature, weight) tuples for the predicted class,
              or None if an error occurs.
    """
    if not raw_text:
        return None

    explainer = LimeTextExplainer(class_names=class_names)

    # LIME's LimeTextExplainer expects a function that takes a list of raw text strings
    # and returns a numpy array of probability scores for each class.
    def predictor(texts):
        if not texts:
            return np.array([])

        # 1. Preprocess texts
        processed_texts = [preprocessor_func(text) for text in texts]

        # 2. Vectorize preprocessed texts
        # Ensure vectorizer is fitted and expects a list of strings
        try:
            vectorized_texts = vectorizer.transform(processed_texts)
        except Exception as e:
            print(f"Error during vectorization in LIME predictor: {e}")
            # Return a dummy probability array that matches expected shape if possible
            # This helps LIME not to crash but indicates an issue.
            # Number of classes:
            num_classes = len(class_names) if class_names is not None else (model.classes_.shape[0] if hasattr(model, 'classes_') else 2)
            return np.full((len(texts), num_classes), 1.0 / num_classes)


        # 3. Get prediction probabilities
        # Ensure the model has a predict_proba method
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(vectorized_texts)
            return probas
        else:
            # If model doesn't have predict_proba, LIME might not work as expected
            # or might require a different setup. For scikit-learn classifiers,
            # predict_proba is standard.
            print("Warning: Model does not have predict_proba method. LIME explanations might be suboptimal or fail.")
            # Fallback: try to return dummy probabilities
            num_classes = len(class_names) if class_names is not None else (model.classes_.shape[0] if hasattr(model, 'classes_') else 2)
            # Create one-hot encoded predictions if only predict is available (less ideal for LIME)
            predictions = model.predict(vectorized_texts)
            dummy_probas = np.zeros((len(texts), num_classes))
            for i, p in enumerate(predictions):
                if p < num_classes: # Ensure prediction is a valid index
                    dummy_probas[i, p] = 1.0
                else: # Handle unexpected prediction value
                    dummy_probas[i, 0] = 1.0 # Default to first class
            return dummy_probas

    try:
        # Explain the prediction for the single raw_text instance
        # LIME's explain_instance expects the raw text, not preprocessed.
        # It will use the `predictor` function internally, which handles preprocessing.
        explanation = explainer.explain_instance(
            raw_text,
            predictor,
            num_features=num_features,
            top_labels=1 # Explain only the top predicted label
        )

        # Get the explanation for the top label
        # The explanation.as_list() returns features for a specific label.
        # If top_labels=1, it will be for the predicted class.
        # We need to get the label for which the explanation was generated.
        # LIME's internal prediction might differ if probabilities are very close.
        # It's usually the first label in explanation.local_exp if top_labels=1

        predicted_label_idx = explanation.top_labels[0]
        explanation_list = explanation.as_list(label=predicted_label_idx)

        # Convert to a more serializable format if needed, e.g., dict
        # For now, returning as list of tuples: [(word, weight), ...]
        return explanation_list

    except Exception as e:
        print(f"Error during LIME explanation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # This is a placeholder for testing the LIME explainer.
    # It requires a trained model, vectorizer, and the preprocessor function.

    # --- Mock objects for testing ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    import sys, os
    # Add src directory to allow importing preprocessing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocessing import preprocess_text_spacy, NLP_SPACY # Assuming NLP_SPACY is loaded in preprocessing

    if NLP_SPACY is None:
        print("spaCy model 'en_core_web_sm' not loaded. Cannot run LIME test.")
        print("Please ensure spaCy model is downloaded and loads correctly in preprocessing.py")
    else:
        print("Setting up mock objects for LIME explainer test...")
        # Sample data
        sample_corpus = [
            "This is a wonderfully fantastic movie, I loved it!",
            "This film was terribly boring and a complete waste of time.",
            "The movie was okay, neither good nor bad."
        ]
        sample_labels = [1, 0, 1] # 1 for positive, 0 for negative

        # 1. Preprocess
        processed_corpus = [preprocess_text_spacy(text) for text in sample_corpus]

        if not all(processed_corpus): # Check if any preprocessing failed
             print("Mock LIME test: Preprocessing failed for some texts. Ensure spaCy model is loaded in preprocessing.py.")
        else:
            # 2. Vectorize
            mock_vectorizer = TfidfVectorizer()
            mock_vectorizer.fit(processed_corpus)
            X_mock = mock_vectorizer.transform(processed_corpus)

            # 3. Train a mock model
            mock_model = LogisticRegression()
            mock_model.fit(X_mock, sample_labels)
            print("Mock model trained.")

            # 4. Test LIME explanation
            test_raw_text_positive = "This is an amazing and fantastic piece of art, truly enjoyable."
            test_raw_text_negative = "I absolutely hated this, it's a boring and terrible waste of my time."

            class_names_test = ['negative', 'positive'] # Must match model's understanding of 0 and 1

            print(f"\nExplaining positive text: '{test_raw_text_positive}'")
            explanation_pos = explain_instance_lime(
                test_raw_text_positive,
                mock_model,
                mock_vectorizer,
                preprocess_text_spacy, # Pass the actual preprocessor function
                class_names=class_names_test
            )
            if explanation_pos:
                print("LIME Explanation (Positive):")
                for feature, weight in explanation_pos:
                    print(f"  {feature}: {weight:.4f}")
            else:
                print("Failed to get LIME explanation for positive text.")

            print(f"\nExplaining negative text: '{test_raw_text_negative}'")
            explanation_neg = explain_instance_lime(
                test_raw_text_negative,
                mock_model,
                mock_vectorizer,
                preprocess_text_spacy,
                class_names=class_names_test
            )
            if explanation_neg:
                print("LIME Explanation (Negative):")
                for feature, weight in explanation_neg:
                    print(f"  {feature}: {weight:.4f}")
            else:
                print("Failed to get LIME explanation for negative text.")
