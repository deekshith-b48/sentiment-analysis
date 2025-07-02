# src/models.py
"""
Module for model definition, training, and evaluation.
Includes:
- Logistic Regression
- SVM
- Random Forest
- LSTM/GRU (later)
- Transformers (later)
"""
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # Ensuring GridSearchCV import is present
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import pandas as pd

# Placeholder for more advanced model imports (PyTorch/TensorFlow)
# import torch
# import transformers

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_logistic_regression(X_train, y_train, model_filename="logistic_regression_model.joblib", use_grid_search=False):
    """
    Trains a Logistic Regression model and saves it.
    Optionally performs GridSearchCV for hyperparameter tuning.
    """
    model_path = os.path.join(MODEL_DIR, model_filename)

    if use_grid_search:
        print("Training Logistic Regression model with GridSearchCV...")
        # Define a basic parameter grid
        # Reduced C values and max_iter for faster grid search on potentially large TF-IDF
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['liblinear'], # liblinear is good for smaller datasets and L1/L2 regularization
            'penalty': ['l1', 'l2']
        }
        # Initial model
        lr = LogisticRegression(random_state=42, max_iter=500) # Reduced max_iter for individual fits in grid search

        # GridSearchCV setup - cv=3 for speed
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

        try:
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
            print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")
        except Exception as e:
            print(f"GridSearchCV failed: {e}. Training with default parameters instead.")
            # Fallback to default training if GridSearchCV fails (e.g., due to small dataset issues with cv)
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)

    else:
        print("Training Logistic Regression model with default parameters...")
        model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter for convergence
        model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"Logistic Regression model saved to {model_path}")
    return model

def train_svm(X_train, y_train, model_filename="svm_model.joblib"):
    """
    Trains an SVM model and saves it.
    """
    print("Training SVM model...")
    model = SVC(random_state=42, probability=True) # probability=True for ROC AUC if needed
    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(model, model_path)
    print(f"SVM model saved to {model_path}")
    return model

def train_random_forest(X_train, y_train, model_filename="random_forest_model.joblib"):
    """
    Trains a Random Forest model and saves it.
    """
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(model, model_path)
    print(f"Random Forest model saved to {model_path}")
    return model

def load_model(model_filename="logistic_regression_model.joblib"):
    """
    Loads a saved model.
    """
    model_path = os.path.join(MODEL_DIR, model_filename)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model file {model_path} not found.")
        return None

def evaluate_model(model, X_test, y_test, average_method='weighted'):
    """
    Evaluates the model and returns a dictionary of metrics.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average=average_method, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    # Confusion Matrix
    # cm = confusion_matrix(y_test, predictions)
    # print("Confusion Matrix:")
    # print(cm) # Or return it if needed

    # ROC-AUC (for binary or multiclass if probabilities are available)
    # try:
    #     if hasattr(model, "predict_proba"):
    #         y_pred_proba = model.predict_proba(X_test)
    #         # For binary classification
    #         if y_pred_proba.shape[1] == 2:
    #             metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
    #         # For multiclass (OvR)
    #         # else:
    #         #     metrics["roc_auc_ovr"] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average=average_method)
    # except Exception as e:
    #     print(f"Could not calculate ROC AUC: {e}")

    return metrics

if __name__ == '__main__':
    # This is a placeholder for actual training data and uses the new spaCy preprocessor
    # In a real scenario, you'd load and preprocess data here
    from sklearn.model_selection import train_test_split
    # Assuming src.preprocessing can be found (e.g. if models.py is run from project root)
    try:
        from preprocessing import preprocess_text_spacy, get_tfidf_vectorizer
    except ImportError:
        # Fallback for running directly from src for testing, adjust path if needed
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from preprocessing import preprocess_text_spacy, get_tfidf_vectorizer


    # Sample data (replace with actual data loading)
    texts = [
        "This movie was absolutely fantastic, a true masterpiece!",
        "I hated this film, it was boring and terribly acted.",
        "The product is okay, not great but not bad either.",
        "Amazing experience, would definitely recommend to everyone.",
        "A complete waste of time and money, very disappointing."
    ]
    # Corresponding labels (e.g., 0 for negative, 1 for positive)
    labels = [1, 0, 1, 1, 0]

    print("Preprocessing sample data with spaCy...")
    # Ensure spaCy model is loaded if NLP_SPACY is None in preprocessing.py
    # For this test, we assume it gets loaded.
    processed_texts = [preprocess_text_spacy(text) for text in texts]
    if any(not t for t in processed_texts): # Check if any text failed preprocessing (e.g. spaCy model not loaded)
        print("Warning: Some texts failed preprocessing, possibly due to spaCy model not being loaded.")
        print("Skipping model training tests in models.py if NLP_SPACY is not available.")
    else:
        print("Vectorizing text data with TF-IDF...")
        tfidf_vectorizer = get_tfidf_vectorizer(processed_texts, max_features=100)
    X = tfidf_vectorizer.transform(processed_texts)
    y = pd.Series(labels) # Using pandas Series for y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    if lr_model:
        lr_metrics = evaluate_model(lr_model, X_test, y_test)
        print("\nLogistic Regression Metrics:")
        for metric, value in lr_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

    # Train SVM
    # svm_model = train_svm(X_train, y_train)
    # if svm_model:
    #     svm_metrics = evaluate_model(svm_model, X_test, y_test)
    #     print("\nSVM Metrics:")
    #     for metric, value in svm_metrics.items():
    #         print(f"{metric.capitalize()}: {value:.4f}")

    # Train Random Forest
    # rf_model = train_random_forest(X_train, y_train)
    # if rf_model:
    #     rf_metrics = evaluate_model(rf_model, X_test, y_test)
    #     print("\nRandom Forest Metrics:")
    #     for metric, value in rf_metrics.items():
    #         print(f"{metric.capitalize()}: {value:.4f}")

    # Example of loading a model
    loaded_lr_model = load_model("logistic_regression_model.joblib")
    if loaded_lr_model:
        print("\nTesting loaded Logistic Regression model:")
        loaded_metrics = evaluate_model(loaded_lr_model, X_test, y_test)
        for metric, value in loaded_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

    # Note: For more complex models like LSTMs or Transformers,
    # the training and evaluation pipeline will be significantly different,
    # likely involving PyTorch or TensorFlow.
    # This initial setup focuses on scikit-learn compatible models.
