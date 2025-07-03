# src/utils.py
"""
Utility functions for the project.

This module includes helpers for:
- Loading and saving JSON configurations.
- Loading datasets from CSV files.
- Calculating and printing detailed classification metrics.
"""

import json
import os
from typing import Any, Dict, List, Tuple, Optional, Union, Sequence

import pandas as pd
import numpy as np # For type hinting arrays
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Loads a JSON configuration file.

    Args:
        config_path (str, optional): Path to the JSON configuration file.
            Defaults to "config.json".

    Returns:
        Dict[str, Any]: A dictionary containing the configuration data.
                        Returns an empty dictionary if the file is not found or
                        if there's an error parsing the JSON.
    """
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Returning empty dictionary.")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config: Dict[str, Any] = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {config_path}: {e}. Returning empty dictionary.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while loading config {config_path}: {e}. Returning empty dictionary.")
        return {}

def save_config(config_data: Dict[str, Any], config_path: str = "config.json") -> None:
    """
    Saves data to a JSON configuration file.

    Args:
        config_data (Dict[str, Any]): Dictionary containing the configuration to save.
        config_path (str, optional): Path to save the JSON configuration file.
            Defaults to "config.json".
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"An error occurred while saving config to {config_path}: {e}")


def load_dataset_csv(
    file_path: str,
    text_column: str,
    label_column: str,
    encoding: str = 'utf-8',
    **kwargs: Any
) -> Tuple[Optional[List[str]], Optional[List[Any]]]:
    """
    Loads a dataset from a CSV file into lists of texts and labels.

    Args:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column containing text data.
        label_column (str): Name of the column containing labels.
        encoding (str, optional): File encoding. Defaults to 'utf-8'.
        **kwargs (Any): Additional arguments to pass to `pd.read_csv`.

    Returns:
        Tuple[Optional[List[str]], Optional[List[Any]]]: A tuple containing:
            - A list of text strings.
            - A list of labels.
            Returns (None, None) if an error occurs during loading or processing.
    """
    try:
        df: pd.DataFrame = pd.read_csv(file_path, encoding=encoding, **kwargs)
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV at {file_path}.")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV at {file_path}.")

        texts: List[str] = df[text_column].astype(str).tolist() # Ensure texts are strings
        labels: List[Any] = df[label_column].tolist()
        print(f"Dataset loaded from {file_path}. Found {len(texts)} samples.")
        return texts, labels
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}.")
        return None, None
    except ValueError as ve: # Catch specific errors like column not found
        print(f"ValueError during CSV loading: {ve}")
        return None, None
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return None, None

MetricsDict = Dict[str, Union[float, int, List[List[int]]]] # Type alias for metrics dictionary

def calculate_detailed_metrics(
    y_true: Union[np.ndarray, List[Any], pd.Series],
    y_pred: Union[np.ndarray, List[Any], pd.Series],
    y_proba: Optional[Union[np.ndarray, List[List[float]]]] = None,
    class_labels: Optional[Sequence[str]] = None,
    average_method: str = 'weighted'
) -> MetricsDict:
    """
    Calculates and prints detailed classification metrics.

    Includes overall accuracy, precision, recall, F1-score (averaged),
    per-class precision, recall, F1-score, support, confusion matrix,
    and ROC AUC score if probabilities are provided.

    Args:
        y_true (Union[np.ndarray, List[Any], pd.Series]): True labels.
        y_pred (Union[np.ndarray, List[Any], pd.Series]): Predicted labels.
        y_proba (Optional[Union[np.ndarray, List[List[float]]]], optional):
            Predicted probabilities (n_samples, n_classes). Required for ROC AUC.
            Defaults to None.
        class_labels (Optional[Sequence[str]], optional): Ordered list of class names for display
            in per-class metrics. If None, integer labels are used. Defaults to None.
        average_method (str, optional): Averaging method for precision, recall, F1 for multiclass.
            Options: 'weighted', 'micro', 'macro'. Defaults to 'weighted'.

    Returns:
        MetricsDict: A dictionary containing the calculated metrics.
                     Keys include 'accuracy', 'precision', 'recall', 'f1_score',
                     per-class metrics (e.g., 'precision_class_positive'),
                     'confusion_matrix', and 'roc_auc' (if applicable).
    """
    # Ensure inputs are numpy arrays for scikit-learn compatibility if they are lists
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    accuracy: float = accuracy_score(y_true_np, y_pred_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average=average_method, zero_division=0
    )

    metrics: MetricsDict = {
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
    unique_labels_in_y_true = sorted(list(pd.unique(y_true_np))) # Get unique labels present in y_true

    # Determine labels for per-class calculation: use provided class_labels indices or unique_labels_in_y_true
    # This ensures that `labels` argument in precision_recall_fscore_support matches the actual classes
    # and their order if class_labels are provided.
    report_labels_indices = list(range(len(class_labels))) if class_labels else unique_labels_in_y_true

    if class_labels:
        display_names = class_labels
    else: # Create display names if not provided
        display_names = [f"Class {lbl}" for lbl in report_labels_indices]

    if len(report_labels_indices) > 0:
        prec_per_class, rec_per_class, f1_per_class, sup_per_class = precision_recall_fscore_support(
            y_true_np, y_pred_np, labels=report_labels_indices, zero_division=0
        )
        print("\n--- Per-class Metrics ---")
        for i, label_idx in enumerate(report_labels_indices):
            current_display_name = display_names[i]
            print(f"Class: {current_display_name} (Label Index: {label_idx})")
            print(f"  Precision: {prec_per_class[i]:.4f}")
            print(f"  Recall:    {rec_per_class[i]:.4f}")
            print(f"  F1-score:  {f1_per_class[i]:.4f}")
            print(f"  Support:   {sup_per_class[i]}")
            metrics[f"precision_class_{current_display_name.replace(' ', '_').lower()}"] = prec_per_class[i]
            metrics[f"recall_class_{current_display_name.replace(' ', '_').lower()}"] = rec_per_class[i]
            metrics[f"f1_score_class_{current_display_name.replace(' ', '_').lower()}"] = f1_per_class[i]
            metrics[f"support_class_{current_display_name.replace(' ', '_').lower()}"] = int(sup_per_class[i])


    cm: np.ndarray = confusion_matrix(y_true_np, y_pred_np, labels=unique_labels_in_y_true if not class_labels else report_labels_indices)
    print("\nConfusion Matrix:")
    print(cm)
    metrics["confusion_matrix"] = cm.tolist()

    if y_proba is not None:
        y_proba_np = np.asarray(y_proba)
        num_classes_from_proba = y_proba_np.shape[1]

        try:
            if num_classes_from_proba == 2 and len(unique_labels_in_y_true) <= 2 : # Binary classification
                # For binary, roc_auc_score expects probabilities of the positive class
                roc_auc: float = roc_auc_score(y_true_np, y_proba_np[:, 1])
                print(f"ROC AUC Score: {roc_auc:.4f}")
                metrics["roc_auc"] = roc_auc
            elif num_classes_from_proba > 2 and len(unique_labels_in_y_true) > 2: # Multiclass
                roc_auc_ovr: float = roc_auc_score(y_true_np, y_proba_np, multi_class='ovr', average=average_method, labels=unique_labels_in_y_true)
                print(f"ROC AUC Score (OvR, {average_method}): {roc_auc_ovr:.4f}")
                metrics[f"roc_auc_ovr_{average_method}"] = roc_auc_ovr
            # else:
            #     print("ROC AUC not calculated: Inconsistent number of classes or probabilities shape.")
        except ValueError as e:
            print(f"Could not calculate ROC AUC score: {e}. Ensure y_proba has correct shape and y_true contains relevant classes.")
        except Exception as e:
            print(f"An error occurred during ROC AUC calculation: {e}")

    return metrics

if __name__ == '__main__':
    # Example usage of configuration functions
    print("--- Config Management Example ---")
    dummy_config_data: Dict[str, Any] = {"model_type": "LogisticRegression", "max_features": 5000, "version": 1.0}
    test_config_path: str = "dummy_test_config.json"
    save_config(dummy_config_data, test_config_path)
    loaded_config_data: Dict[str, Any] = load_config(test_config_path)
    print(f"Loaded config: {loaded_config_data}")
    if os.path.exists(test_config_path):
        os.remove(test_config_path)

    # Example usage of load_dataset_csv
    print("\n--- CSV Loading Example ---")
    dummy_csv_data: Dict[str, List[Any]] = {
        'review_text': ["Great movie!", "Bad film.", "Okay, I guess.", "Superb acting."],
        'sentiment_label': ["positive", "negative", "neutral", "positive"]
    }
    dummy_df_main: pd.DataFrame = pd.DataFrame(dummy_csv_data)
    test_csv_path: str = "dummy_test_data.csv"
    dummy_df_main.to_csv(test_csv_path, index=False)

    texts_main, labels_main = load_dataset_csv(test_csv_path, "review_text", "sentiment_label")
    if texts_main and labels_main:
        print(f"Texts from CSV: {texts_main}")
        print(f"Labels from CSV: {labels_main}")
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)

    # Example usage of calculate_detailed_metrics
    print("\n--- Detailed Metrics Example (Multiclass) ---")
    y_true_sample_mc: List[int] = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2]
    y_pred_sample_mc: List[int] = [0, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1]
    y_proba_sample_mc: List[List[float]] = [
        [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.6, 0.3], [0.9, 0.05, 0.05],
        [0.1, 0.3, 0.6], [0.05, 0.15, 0.8], [0.7, 0.2, 0.1], [0.4, 0.5, 0.1],
        [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.2, 0.5, 0.3]
    ]
    class_names_mc: List[str] = ["negative", "neutral", "positive"]

    detailed_metrics_mc = calculate_detailed_metrics(
        y_true_sample_mc, y_pred_sample_mc,
        y_proba=y_proba_sample_mc,
        class_labels=class_names_mc
    )
    # print(f"Returned multiclass metrics dictionary: {detailed_metrics_mc}")

    print("\n--- Detailed Metrics Example (Binary) ---")
    y_true_binary_main: List[int] = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0]
    y_pred_binary_main: List[int] = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]
    y_proba_binary_main: List[List[float]] = [
        [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4], [0.95, 0.05],
        [0.1, 0.9], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7], [0.15, 0.85], [0.85, 0.15]
    ]
    binary_class_names_main: List[str] = ["negative", "positive"]
    detailed_binary_metrics_main = calculate_detailed_metrics(
        y_true_binary_main, y_pred_binary_main,
        y_proba=y_proba_binary_main,
        class_labels=binary_class_names_main
    )
    # print(f"Returned binary metrics dictionary: {detailed_binary_metrics_main}")
```
