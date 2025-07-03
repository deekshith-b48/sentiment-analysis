# train_model.py
"""
Main script to train a sentiment analysis model.

This script handles the end-to-end process of:
1. Loading a dataset (currently IMDb from Hugging Face datasets).
2. Subsetting the data if specified for quick runs.
3. Preprocessing text data using spaCy.
4. Splitting data into training and evaluation sets.
5. Vectorizing text data using TF-IDF.
6. Training a specified machine learning model (Logistic Regression, SVM, Random Forest).
   - Optionally includes hyperparameter tuning for Logistic Regression via GridSearchCV.
7. Evaluating the trained model on the evaluation set.
8. Saving the trained model and TF-IDF vectorizer to disk.
"""

import argparse
import os
import joblib
from typing import List, Dict, Any, Tuple, Union

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset

# Ensure src modules can be imported
import sys
current_dir_path: str = os.path.dirname(os.path.abspath(__file__))
src_path: str = os.path.join(current_dir_path, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from preprocessing import preprocess_text_spacy, get_tfidf_vectorizer
from models import (
    train_logistic_regression,
    train_svm,
    train_random_forest,
    MODEL_DIR,
    SklearnModel # Assuming SklearnModel type hint is defined in models.py
)
from utils import calculate_detailed_metrics, MetricsDict

# Define default sentiment labels for IMDb (0: negative, 1: positive)
IMDB_SENTIMENT_LABELS: Dict[int, str] = {0: "negative", 1: "positive"}


def main(args: argparse.Namespace) -> None:
    """
    Executes the main model training and evaluation pipeline based on provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    print("Starting model training process...")

    # --- 1. Load Dataset ---
    print("Loading IMDb dataset...")
    try:
        imdb_dataset: DatasetDict = load_dataset("imdb")

        # Extract full original training and test sets
        train_dataset_full: Dataset = imdb_dataset['train']
        test_dataset_full: Dataset = imdb_dataset['test']

        train_texts_orig: List[str] = [item['text'] for item in train_dataset_full]
        train_labels_orig: List[int] = [item['label'] for item in train_dataset_full]

        test_texts_orig: List[str] = [item['text'] for item in test_dataset_full]
        test_labels_orig: List[int] = [item['label'] for item in test_dataset_full]

        print(f"Loaded {len(train_texts_orig)} original training samples and {len(test_texts_orig)} original test samples.")

        # Initialize variables for texts and labels to be used in model training and evaluation
        X_train_model_texts: List[str]
        y_train_model_labels: List[int]
        X_test_model_eval_texts: List[str]
        y_test_model_eval_labels: List[int]

        if args.data_subset > 0:
            subset_size: int = min(args.data_subset, len(train_texts_orig))
            if subset_size < 20: # Minimum for reliable stratification and splitting
                print(f"Error: Data subset size of {subset_size} is too small. Minimum 20 recommended for stratification.")
                return

            print(f"Creating a stratified subset of {subset_size} samples from the original training data.")
            # Create a working subset from the original training data
            working_texts_subset, _, working_labels_subset, _ = train_test_split(
                train_texts_orig, train_labels_orig,
                train_size=subset_size,
                stratify=train_labels_orig,
                random_state=42 # For reproducibility
            )

            # Split this working subset into training and a temporary evaluation set for this script run
            # Ensure test_size is reasonable (e.g., 0.25 implies at least 4 samples in working_labels_subset for stratification)
            test_split_size = 0.25
            if len(working_labels_subset) * test_split_size < 2 * len(np.unique(working_labels_subset)): # Ensure enough samples per class for test
                print(f"Warning: Subset too small for a {test_split_size*100}% test split with stratification. Adjusting test size or using full subset for training.")
                # Fallback: use a smaller test split or just train on the whole working_texts_subset and evaluate on original test set.
                # For simplicity here, if too small for a robust test split from subset, we might just use the whole subset for training
                # and rely on the original test set for evaluation if the user wants a quick run.
                # However, the current logic aims to provide a quick evaluation on a subset-derived test set.
                # Let's ensure at least a minimal test set if possible.
                if len(working_labels_subset) > 4 : # Need at least some samples to split
                     X_train_model_texts, X_test_model_eval_texts, y_train_model_labels, y_test_model_eval_labels = train_test_split(
                        working_texts_subset, working_labels_subset,
                        test_size=test_split_size,
                        stratify=working_labels_subset, # Stratify on the subset labels
                        random_state=42
                    )
                else: # Not enough to split, use all for training, and original test set for eval
                    print("Subset too small to create a distinct evaluation set from it. Training on entire subset.")
                    X_train_model_texts, y_train_model_labels = working_texts_subset, working_labels_subset
                    X_test_model_eval_texts, y_test_model_eval_labels = test_texts_orig, test_labels_orig # Fallback to original test set

            else:
                 X_train_model_texts, X_test_model_eval_texts, y_train_model_labels, y_test_model_eval_labels = train_test_split(
                    working_texts_subset, working_labels_subset,
                    test_size=test_split_size,
                    stratify=working_labels_subset,
                    random_state=42
                )
            print(f"Using subset for this run: {len(X_train_model_texts)} training, {len(X_test_model_eval_texts)} for script evaluation.")

        else: # Use full original training and test sets
            X_train_model_texts = train_texts_orig
            y_train_model_labels = train_labels_orig
            X_test_model_eval_texts = test_texts_orig
            y_test_model_eval_labels = test_labels_orig
            print(f"Using full dataset: {len(X_train_model_texts)} training, {len(X_test_model_eval_texts)} for script evaluation.")

        if not X_train_model_texts or not y_train_model_labels:
            print("Error: Model training set is empty after subsetting/splitting.")
            return
        if not X_test_model_eval_texts or not y_test_model_eval_labels:
            print("Error: Model evaluation set for this script run is empty.")
            return

    except Exception as e:
        print(f"Error loading or splitting dataset: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure you have an internet connection and the 'datasets' library is installed.")
        return

    # --- 2. Preprocess Text Data ---
    print("Preprocessing text data (this may take some time)...")
    X_train_processed: List[str] = [preprocess_text_spacy(text) for text in X_train_model_texts]
    X_test_processed: List[str] = [preprocess_text_spacy(text) for text in X_test_model_eval_texts]
    print("Preprocessing complete.")

    # --- 3. Vectorize Text Data (TF-IDF) ---
    print(f"Vectorizing text data with TF-IDF (max_features={args.max_features})...")
    tfidf_vectorizer: TfidfVectorizer = get_tfidf_vectorizer(texts=X_train_processed, max_features=args.max_features)

    X_train_tfidf: spmatrix = tfidf_vectorizer.transform(X_train_processed)
    X_test_tfidf: spmatrix = tfidf_vectorizer.transform(X_test_processed)
    print("Vectorization complete.")
    print(f"TF-IDF training matrix shape: {X_train_tfidf.shape}")
    print(f"TF-IDF test matrix shape: {X_test_tfidf.shape}")

    vectorizer_filename: str = args.vectorizer_file
    vectorizer_path: str = os.path.join(MODEL_DIR, vectorizer_filename)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")

    y_train: List[int] = y_train_model_labels
    y_test: List[int] = y_test_model_eval_labels

    # --- 4. Train Model ---
    model_filename_arg: str = args.model_file
    # Adjust default model filename based on model type if user didn't specify a custom one
    final_model_filename: str = model_filename_arg
    if model_filename_arg == "logistic_regression_model.joblib": # Default from argparse
        final_model_filename = f"{args.model_type}_model.joblib"

    model: Optional[SklearnModel] = None
    if args.model_type == "logistic_regression":
        model = train_logistic_regression(
            X_train_tfidf, y_train,
            model_filename=final_model_filename,
            use_grid_search=args.tune_hyperparameters
        )
    elif args.model_type == "svm":
        model = train_svm(X_train_tfidf, y_train, model_filename=final_model_filename)
    elif args.model_type == "random_forest":
        model = train_random_forest(X_train_tfidf, y_train, model_filename=final_model_filename)
    else:
        # Should not be reached due to argparse choices
        print(f"FATAL: Unsupported model type specified: {args.model_type}")
        return

    if model is None:
        print(f"Model training failed for {args.model_type}.")
        return
    print(f"{args.model_type} model trained successfully and saved as {final_model_filename}.")

    # --- 5. Evaluate Model ---
    print(f"Evaluating {args.model_type} model on the evaluation set...")

    y_pred: np.ndarray = model.predict(X_test_tfidf)
    y_proba: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test_tfidf)
        except Exception as e_proba:
            print(f"Could not get probabilities from model: {e_proba}")

    # Ensure class names are correctly ordered for metrics calculation
    # For IMDb, labels are 0 (neg) and 1 (pos).
    class_names_for_metrics: List[str] = [IMDB_SENTIMENT_LABELS[i] for i in sorted(IMDB_SENTIMENT_LABELS.keys())]

    metrics: MetricsDict = calculate_detailed_metrics(y_test, y_pred, y_proba, class_labels=class_names_for_metrics)
    print("\n--- Detailed Test Set Evaluation Summary ---")
    for metric_name, value in metrics.items():
        if not isinstance(value, list): # Avoid printing full confusion matrix list again
             print(f"{metric_name.replace('_', ' ').capitalize()}: {value}")

    print(f"\nTraining complete. Model '{final_model_filename}' and vectorizer '{vectorizer_filename}' saved in '{MODEL_DIR}'.")
    print(f"To use this specific model with the API, ensure MODEL_FILENAME in 'api/app.py' is updated to '{final_model_filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a sentiment analysis model on the IMDb dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "svm", "random_forest"],
        help="Type of model to train."
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Maximum number of features for TF-IDF vectorizer."
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="logistic_regression_model.joblib", # Default, will be adapted by model_type
        help="Filename for saving the trained model in the 'trained_models' directory."
    )
    parser.add_argument(
        "--vectorizer_file",
        type=str,
        default="tfidf_vectorizer.joblib",
        help="Filename for saving the TF-IDF vectorizer in the 'trained_models' directory."
    )
    parser.add_argument(
        "--data_subset",
        type=int,
        default=0,
        help="Number of samples from the original training data to use for a quick run (0 for full dataset). "
             "A portion of this subset will be used for training and another for temporary evaluation."
    )
    parser.add_argument(
        "--tune_hyperparameters",
        action="store_true",
        help="Enable hyperparameter tuning (GridSearchCV) for supported models (currently Logistic Regression)."
    )

    cli_args: argparse.Namespace = parser.parse_args()
    main(cli_args)

    # Example commands:
    # python train_model.py --model_type logistic_regression --data_subset 1000 --tune_hyperparameters
    # python train_model.py --model_type svm --data_subset 1000
    # python train_model.py --model_type random_forest
```
