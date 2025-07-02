# train_model.py
"""
Script to train a sentiment analysis model.
Steps:
1. Load dataset (e.g., IMDb from Hugging Face datasets).
2. Preprocess text data.
3. Split data into training and test sets.
4. Vectorize text data using TF-IDF.
5. Train a specified model (e.g., Logistic Regression).
6. Evaluate the model.
7. Save the trained model and vectorizer.
"""

import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Ensure src modules can be imported
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming train_model.py is in the root, and src is a subdirectory
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from preprocessing import preprocess_text_spacy, get_tfidf_vectorizer # Changed to spaCy
from models import train_logistic_regression, evaluate_model, MODEL_DIR
from utils import calculate_detailed_metrics

# Define default sentiment labels for IMDb (0: negative, 1: positive)
IMDB_SENTIMENT_LABELS = {0: "negative", 1: "positive"}


def main(args):
    """Main training pipeline."""
    print("Starting model training process...")

    # --- 1. Load Dataset ---
    print("Loading IMDb dataset...")
    try:
        # Using the 'imdb' dataset from Hugging Face datasets library
        imdb_dataset = load_dataset("imdb")

        train_texts_orig = [item['text'] for item in imdb_dataset['train']]
        train_labels_orig = [item['label'] for item in imdb_dataset['train']]

        test_texts_orig = [item['text'] for item in imdb_dataset['test']]
        test_labels_orig = [item['label'] for item in imdb_dataset['test']]

        print(f"Loaded {len(train_texts_orig)} original training samples and {len(test_texts_orig)} original test samples.")

        # These will be the actual texts and labels used for training the model and for the evaluation within this script.
        # If using a subset, they will be derived from train_texts_orig.
        # If not using a subset, X_train_model will be train_texts_orig, and X_test_model_eval will be test_texts_orig.
        X_train_model_texts, y_train_model_labels = [], []
        X_test_model_eval_texts, y_test_model_eval_labels = [], []

        if args.data_subset > 0:
            subset_size = min(args.data_subset, len(train_texts_orig))
            if subset_size < 20: # Need a minimum for stratification
                print("Error: Data subset is too small for reliable stratified splitting. Minimum 20 recommended.")
                return

            print(f"Creating a stratified training subset of {subset_size} samples from the original training data.")
            # Take a subset from the original training data for the model training process
            # This subset will be further split into a train and a (temporary) test set for this run.
            working_texts_subset, _, working_labels_subset, _ = train_test_split(
                train_texts_orig, train_labels_orig,
                train_size=subset_size,
                stratify=train_labels_orig,
                random_state=42
            )

            # Split this working subset into training and a temporary evaluation set for this script run
            X_train_model_texts, X_test_model_eval_texts, y_train_model_labels, y_test_model_eval_labels = train_test_split(
                working_texts_subset, working_labels_subset,
                test_size=0.25, # e.g., 25% of the subset for temporary evaluation
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
            # This could happen if the subset is extremely small.
            print("Error: Model evaluation set for this script run is empty. This might indicate the subset is too small.")
            return # Or handle by skipping evaluation if appropriate

    except Exception as e:
        print(f"Error loading or splitting dataset: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure you have an internet connection and the 'datasets' library is installed.")
        print("You might need to run: pip install datasets")
        return

    # --- 2. Preprocess Text Data ---
    print("Preprocessing text data (this may take some time)...")
    X_train_processed = [preprocess_text_spacy(text) for text in X_train_model_texts]
    X_test_processed = [preprocess_text_spacy(text) for text in X_test_model_eval_texts] # Preprocess the evaluation set
    print("Preprocessing complete.")

    # --- 3. Vectorize Text Data (TF-IDF) ---
    print(f"Vectorizing text data with TF-IDF (max_features={args.max_features})...")
    # Fit TF-IDF only on the training data
    tfidf_vectorizer = get_tfidf_vectorizer(texts=X_train_processed, max_features=args.max_features)

    X_train_tfidf = tfidf_vectorizer.transform(X_train_processed)
    # Transform the test data using the *fitted* vectorizer
    X_test_tfidf = tfidf_vectorizer.transform(X_test_processed)
    print("Vectorization complete.")
    print(f"TF-IDF training matrix shape: {X_train_tfidf.shape}")
    print(f"TF-IDF test matrix shape: {X_test_tfidf.shape}")

    # Save the fitted vectorizer
    vectorizer_filename = args.vectorizer_file
    vectorizer_path = os.path.join(MODEL_DIR, vectorizer_filename)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")

    # Labels for training and evaluation are now y_train_model_labels and y_test_model_eval_labels
    y_train = y_train_model_labels
    y_test = y_test_model_eval_labels

    # --- 4. Train Model ---
    # For now, only Logistic Regression is implemented as the primary example
    if args.model_type == "logistic_regression":
        model_filename = args.model_file or "logistic_regression_model.joblib"
        model = train_logistic_regression(X_train_tfidf, y_train, model_filename=model_filename)
    # Add conditions for other models like SVM, Random Forest here later
    # elif args.model_type == "svm":
    #     model_filename = args.model_file or "svm_model.joblib"
    #     model = train_svm(X_train_tfidf, y_train, model_filename=model_filename)
    # elif args.model_type == "random_forest":
    #     model_filename = args.model_file or "random_forest_model.joblib"
    #     model = train_random_forest(X_train_tfidf, y_train, model_filename=model_filename)
    else:
        print(f"Unsupported model type: {args.model_type}")
        return

    if model is None:
        print("Model training failed.")
        return
    print(f"{args.model_type} model trained successfully.")

    # --- 5. Evaluate Model ---
    print("Evaluating model on the test set...")
    # Basic evaluation from models.py
    # simple_metrics = evaluate_model(model, X_test_tfidf, y_test)
    # print(f"Basic Evaluation Metrics: {simple_metrics}")

    # Detailed evaluation from utils.py
    y_pred = model.predict(X_test_tfidf)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_tfidf)

    class_names = [IMDB_SENTIMENT_LABELS[i] for i in sorted(IMDB_SENTIMENT_LABELS.keys())]

    detailed_metrics_results = calculate_detailed_metrics(y_test, y_pred, y_proba, class_labels=class_names)
    print("\n--- Detailed Test Set Evaluation ---")
    for k, v in detailed_metrics_results.items():
        if not isinstance(v, list): # Don't print confusion matrix list here
             print(f"{k.replace('_', ' ').capitalize()}: {v}")

    print(f"\nTraining complete. Model and vectorizer saved in '{MODEL_DIR}'.")
    print(f"To use the API, ensure '{model_filename}' and '{vectorizer_filename}' are correctly referenced in 'api/app.py'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression"], # Add "svm", "random_forest" later
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
        default="logistic_regression_model.joblib",
        help="Filename for saving the trained model."
    )
    parser.add_argument(
        "--vectorizer_file",
        type=str,
        default="tfidf_vectorizer.joblib",
        help="Filename for saving the TF-IDF vectorizer."
    )
    parser.add_argument(
        "--data_subset",
        type=int,
        default=0, # 0 means use full dataset, otherwise specify number of samples
        help="Number of data samples to use for a quick test run (0 for full dataset). Splits between train/test."
    )
    # Add arguments for data path if not using Hugging Face datasets later

    args = parser.parse_args()
    main(args)

    # Example command to run:
    # python train_model.py --model_type logistic_regression --max_features 5000 --data_subset 2000 # For a quick test
    # python train_model.py # For full run with defaults
