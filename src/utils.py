# src/utils.py
"""
Utility functions for the project.
May include:
- Data loading helpers
- Configuration management
- Specific evaluation metric calculations (if not covered by sklearn)
- Logging setup
"""

import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

def load_config(config_path="config.json"):
    """Loads a JSON configuration file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return {}
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config_data, config_path="config.json"):
    """Saves data to a JSON configuration file."""
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    print(f"Configuration saved to {config_path}")

def load_dataset_csv(file_path, text_column, label_column, encoding='utf-8', **kwargs):
    """
    Loads a dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column containing text data.
        label_column (str): Name of the column containing labels.
        encoding (str): File encoding.
        **kwargs: Additional arguments for pd.read_csv.

    Returns:
        tuple: (list of texts, list of labels) or (None, None) if error.
    """
    try:
        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV.")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV.")

        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        print(f"Dataset loaded from {file_path}. Found {len(texts)} samples.")
        return texts, labels
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return None, None

def calculate_detailed_metrics(y_true, y_pred, y_proba=None, class_labels=None, average_method='weighted'):
    """
    Calculates and prints detailed classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (for ROC AUC).
        class_labels: List of class names for display.
        average_method: Averaging method for precision, recall, f1 ('weighted', 'micro', 'macro').

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average_method, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        f"{average_method}_precision": precision,
        f"{average_method}_recall": recall,
        f"{average_method}_f1_score": f1,
    }

    print(f"\n--- Classification Metrics ({average_method} average) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Per-class metrics
    if class_labels:
        prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(class_labels))), zero_division=0
        )
        print("\n--- Per-class Metrics ---")
        for i, label in enumerate(class_labels):
            print(f"Class: {label}")
            print(f"  Precision: {prec_per_class[i]:.4f}")
            print(f"  Recall:    {rec_per_class[i]:.4f}")
            print(f"  F1-score:  {f1_per_class[i]:.4f}")
            print(f"  Support:   {support_per_class[i]}")
            metrics[f"precision_class_{label}"] = prec_per_class[i]
            metrics[f"recall_class_{label}"] = rec_per_class[i]
            metrics[f"f1_score_class_{label}"] = f1_per_class[i]
            metrics[f"support_class_{label}"] = support_per_class[i]


    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    metrics["confusion_matrix"] = cm.tolist() # Store as list for JSON serialization if needed

    # ROC-AUC
    if y_proba is not None:
        num_classes = len(set(y_true))
        try:
            if num_classes == 2: # Binary classification
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                print(f"ROC AUC Score: {roc_auc:.4f}")
                metrics["roc_auc"] = roc_auc
            elif num_classes > 2: # Multiclass classification
                # OvR (One-vs-Rest)
                roc_auc_ovr = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average_method)
                print(f"ROC AUC Score (OvR, {average_method}): {roc_auc_ovr:.4f}")
                metrics[f"roc_auc_ovr_{average_method}"] = roc_auc_ovr
                # OvO (One-vs-One)
                # roc_auc_ovo = roc_auc_score(y_true, y_proba, multi_class='ovo', average=average_method)
                # print(f"ROC AUC Score (OvO, {average_method}): {roc_auc_ovo:.4f}")
                # metrics[f"roc_auc_ovo_{average_method}"] = roc_auc_ovo
        except ValueError as e:
            print(f"Could not calculate ROC AUC score: {e}. Ensure y_proba has correct shape for multiclass.")
        except Exception as e:
            print(f"An error occurred during ROC AUC calculation: {e}")

    return metrics

if __name__ == '__main__':
    # Example usage of load_config
    # Create a dummy config.json for testing
    dummy_config = {"model_type": "LogisticRegression", "max_features": 5000}
    save_config(dummy_config, "dummy_config.json")
    loaded_config = load_config("dummy_config.json")
    print(f"Loaded config: {loaded_config}")
    if os.path.exists("dummy_config.json"):
        os.remove("dummy_config.json")

    # Example usage of load_dataset_csv
    # Create a dummy data.csv for testing
    dummy_data = {
        'review_text': ["Great movie!", "Bad film.", "Okay, I guess."],
        'sentiment_label': ["positive", "negative", "neutral"]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv("dummy_data.csv", index=False)

    texts, labels = load_dataset_csv("dummy_data.csv", "review_text", "sentiment_label")
    if texts and labels:
        print(f"Texts: {texts}")
        print(f"Labels: {labels}")
    if os.path.exists("dummy_data.csv"):
        os.remove("dummy_data.csv")

    # Example usage of calculate_detailed_metrics
    y_true_sample = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2]
    y_pred_sample = [0, 1, 1, 0, 2, 2, 0, 1, 1, 2]
    # Example probabilities for 3 classes (ensure shape is n_samples, n_classes)
    y_proba_sample = [
        [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.6, 0.3], [0.9, 0.05, 0.05],
        [0.1, 0.3, 0.6], [0.05, 0.15, 0.8], [0.7, 0.2, 0.1], [0.4, 0.5, 0.1],
        [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]
    ]
    class_names = ["negative", "neutral", "positive"]

    print("\n--- Detailed Metrics Example ---")
    detailed_metrics = calculate_detailed_metrics(y_true_sample, y_pred_sample, y_proba=pd.DataFrame(y_proba_sample).values, class_labels=class_names)
    # print(f"Returned metrics dictionary: {detailed_metrics}")

    print("\n--- Binary Classification Example ---")
    y_true_binary = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    y_pred_binary = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
    y_proba_binary = [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4],
                      [0.95, 0.05], [0.1, 0.9], [0.7, 0.3], [0.4, 0.6],
                      [0.3, 0.7], [0.15, 0.85]]
    binary_class_names = ["negative", "positive"]
    detailed_binary_metrics = calculate_detailed_metrics(y_true_binary, y_pred_binary, y_proba=pd.DataFrame(y_proba_binary).values, class_labels=binary_class_names)
    # print(f"Returned binary metrics: {detailed_binary_metrics}")
