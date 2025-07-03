# src/models.py
"""
Module for model definition, training, evaluation, and loading.

Supports Logistic Regression, SVM, and Random Forest classifiers.
Includes functionality for optional hyperparameter tuning using GridSearchCV
for Logistic Regression.
"""
import joblib
import os
from typing import Any, Dict, Optional, Union, List

import pandas as pd
import numpy as np # For type hinting arrays
from scipy.sparse import spmatrix # For type hinting sparse matrices from TF-IDF

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import confusion_matrix, roc_auc_score # Retained for potential future use in evaluate_model

# Define a more specific type for scikit-learn model objects if desired,
# otherwise Any or a Union of specific models can be used.
SklearnModel = Any # Or Union[LogisticRegression, SVC, RandomForestClassifier, GridSearchCV]

MODEL_DIR: str = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_logistic_regression(
    X_train: Union[np.ndarray, spmatrix],
    y_train: Union[np.ndarray, pd.Series],
    model_filename: str = "logistic_regression_model.joblib",
    use_grid_search: bool = False
) -> Optional[SklearnModel]:
    """
    Trains a Logistic Regression model and saves it to disk.

    Optionally performs GridSearchCV for hyperparameter tuning if use_grid_search is True.

    Args:
        X_train (Union[np.ndarray, spmatrix]): Training features (e.g., TF-IDF matrix).
        y_train (Union[np.ndarray, pd.Series]): Training labels.
        model_filename (str, optional): Filename to save the trained model.
            Defaults to "logistic_regression_model.joblib".
        use_grid_search (bool, optional): Whether to perform GridSearchCV.
            Defaults to False.

    Returns:
        Optional[SklearnModel]: The trained scikit-learn model object (LogisticRegression or
                                the best_estimator_ from GridSearchCV), or None if training fails.
    """
    model_path: str = os.path.join(MODEL_DIR, model_filename)
    model: Optional[SklearnModel] = None

    if use_grid_search:
        print("Training Logistic Regression model with GridSearchCV...")
        param_grid: Dict[str, List[Any]] = {
            'C': [0.1, 1, 10],
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2']
        }
        lr = LogisticRegression(random_state=42, max_iter=500) # Base estimator for grid search

        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1 # Use all available cores
        )

        try:
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
            print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")
        except Exception as e:
            print(f"GridSearchCV failed: {e}. Training with default parameters instead.")
            # Fallback to default training
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
    else:
        print("Training Logistic Regression model with default parameters...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

    if model:
        joblib.dump(model, model_path)
        print(f"Logistic Regression model saved to {model_path}")
    return model

def train_svm(
    X_train: Union[np.ndarray, spmatrix],
    y_train: Union[np.ndarray, pd.Series],
    model_filename: str = "svm_model.joblib"
) -> Optional[SVC]:
    """
    Trains an SVM (Support Vector Classifier) model and saves it.

    Args:
        X_train (Union[np.ndarray, spmatrix]): Training features.
        y_train (Union[np.ndarray, pd.Series]): Training labels.
        model_filename (str, optional): Filename for saving the model.
            Defaults to "svm_model.joblib".

    Returns:
        Optional[SVC]: The trained SVC model object, or None if training fails.
    """
    print("Training SVM model...")
    model = SVC(random_state=42, probability=True) # probability=True for ROC AUC if needed later
    try:
        model.fit(X_train, y_train)
        model_path: str = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"SVM model saved to {model_path}")
        return model
    except Exception as e:
        print(f"SVM training failed: {e}")
        return None

def train_random_forest(
    X_train: Union[np.ndarray, spmatrix],
    y_train: Union[np.ndarray, pd.Series],
    model_filename: str = "random_forest_model.joblib"
) -> Optional[RandomForestClassifier]:
    """
    Trains a Random Forest classifier model and saves it.

    Args:
        X_train (Union[np.ndarray, spmatrix]): Training features.
        y_train (Union[np.ndarray, pd.Series]): Training labels.
        model_filename (str, optional): Filename for saving the model.
            Defaults to "random_forest_model.joblib".

    Returns:
        Optional[RandomForestClassifier]: The trained RandomForestClassifier model object,
                                           or None if training fails.
    """
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    try:
        model.fit(X_train, y_train)
        model_path: str = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"Random Forest model saved to {model_path}")
        return model
    except Exception as e:
        print(f"Random Forest training failed: {e}")
        return None

def load_model(model_filename: str = "logistic_regression_model.joblib") -> Optional[SklearnModel]:
    """
    Loads a pre-trained scikit-learn model from disk.

    Args:
        model_filename (str, optional): The filename of the model to load from the `MODEL_DIR`.
            Defaults to "logistic_regression_model.joblib".

    Returns:
        Optional[SklearnModel]: The loaded scikit-learn model object, or None if the file is not found.
    """
    model_path: str = os.path.join(MODEL_DIR, model_filename)
    if os.path.exists(model_path):
        try:
            model: SklearnModel = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
    else:
        print(f"Model file {model_path} not found.")
        return None

def evaluate_model(
    model: SklearnModel,
    X_test: Union[np.ndarray, spmatrix],
    y_test: Union[np.ndarray, pd.Series],
    average_method: str = 'weighted'
) -> Dict[str, float]:
    """
    Evaluates a trained model on a test set.

    Calculates accuracy, precision, recall, and F1-score.
    Other metrics like confusion matrix or ROC-AUC can be added here if needed.

    Args:
        model (SklearnModel): The trained scikit-learn model to evaluate.
        X_test (Union[np.ndarray, spmatrix]): Test features.
        y_test (Union[np.ndarray, pd.Series]): True test labels.
        average_method (str, optional): The averaging method for precision, recall, F1-score
            for multiclass targets. Common options: 'weighted', 'macro', 'micro'.
            Defaults to 'weighted'.

    Returns:
        Dict[str, float]: A dictionary containing the calculated metrics:
                          'accuracy', 'precision', 'recall', 'f1_score'.
                          Returns empty dict if evaluation fails.
    """
    try:
        predictions: np.ndarray = model.predict(X_test)

        accuracy: float = accuracy_score(y_test, predictions)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, predictions, average=average_method, zero_division=0
        )

        metrics: Dict[str, float] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
        return metrics
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return {}

if __name__ == '__main__':
    # This block is for example usage and basic testing of the functions in this module.
    # It relies on preprocess_text_spacy from preprocessing.py.

    from sklearn.model_selection import train_test_split

    # Attempt to import preprocessing functions
    # This allows running 'python src/models.py' from the project root
    try:
        from preprocessing import preprocess_text_spacy, get_tfidf_vectorizer, NLP_SPACY
    except ImportError:
        # Fallback for running directly from src or if path issues occur
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from preprocessing import preprocess_text_spacy, get_tfidf_vectorizer, NLP_SPACY

    if NLP_SPACY is None:
        print("spaCy model NLP_SPACY is not loaded (from preprocessing.py).")
        print("Skipping __main__ example in models.py.")
    else:
        sample_texts: List[str] = [
            "This movie was absolutely fantastic, a true masterpiece!",
            "I hated this film, it was boring and terribly acted.",
            "The product is okay, not great but not bad either.",
            "Amazing experience, would definitely recommend to everyone.",
            "A complete waste of time and money, very disappointing."
        ]
        sample_labels: List[int] = [1, 0, 1, 1, 0]  # 1: positive, 0: negative

        print("Preprocessing sample data with spaCy for models.py test...")
        processed_texts: List[str] = [preprocess_text_spacy(text) for text in sample_texts]

        if not all(t is not None and t != "" for t in processed_texts):
            print("Warning: Some texts failed preprocessing in models.py test.")
        else:
            print("Vectorizing text data with TF-IDF for models.py test...")
            tfidf_vectorizer_example = get_tfidf_vectorizer(processed_texts, max_features=100)
            X_sample: spmatrix = tfidf_vectorizer_example.transform(processed_texts)
            y_sample: pd.Series = pd.Series(sample_labels)

            X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
                X_sample, y_sample, test_size=0.4, random_state=42, stratify=y_sample # Ensure stratification for small sample
            )

            print(f"\nSample training data shape: {X_train_sample.shape}")
            print(f"Sample test data shape: {X_test_sample.shape}")

            # Test Logistic Regression
            print("\n--- Testing Logistic Regression ---")
            lr_model_test = train_logistic_regression(
                X_train_sample, y_train_sample,
                model_filename="lr_test_model.joblib",
                use_grid_search=False # Keep test fast
            )
            if lr_model_test:
                lr_metrics_test = evaluate_model(lr_model_test, X_test_sample, y_test_sample)
                print("Logistic Regression (Test) Metrics:", lr_metrics_test)
                loaded_lr = load_model("lr_test_model.joblib")
                if loaded_lr:
                    print("Successfully loaded lr_test_model.joblib")


            # Test SVM
            print("\n--- Testing SVM ---")
            svm_model_test = train_svm(
                X_train_sample, y_train_sample,
                model_filename="svm_test_model.joblib"
            )
            if svm_model_test:
                svm_metrics_test = evaluate_model(svm_model_test, X_test_sample, y_test_sample)
                print("SVM (Test) Metrics:", svm_metrics_test)

            # Test Random Forest
            print("\n--- Testing Random Forest ---")
            rf_model_test = train_random_forest(
                X_train_sample, y_train_sample,
                model_filename="rf_test_model.joblib"
            )
            if rf_model_test:
                rf_metrics_test = evaluate_model(rf_model_test, X_test_sample, y_test_sample)
                print("Random Forest (Test) Metrics:", rf_metrics_test)
```
