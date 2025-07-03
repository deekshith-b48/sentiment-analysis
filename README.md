# Advanced Sentiment Analysis System

This project provides an end-to-end Machine Learning-based Sentiment Analysis system. It classifies text into positive or negative sentiment, includes model interpretability with LIME, and supports multiple classifier backbones.

## Features

*   **Core Sentiment Classification**: Classifies English text into positive or negative sentiment.
*   **spaCy-based Preprocessing**: Utilizes spaCy for robust tokenization, lemmatization, and stopword removal.
*   **Multiple Classifier Support**: Train and use models like Logistic Regression, SVM, or Random Forest.
*   **Hyperparameter Tuning**: Basic GridSearchCV support for Logistic Regression.
*   **Model Interpretability with LIME**: Explains individual predictions by highlighting key word contributions.
*   **RESTful API**: FastAPI backend to serve real-time sentiment predictions with explanations.
*   **Interactive Dashboard**: Streamlit dashboard for single text analysis and batch CSV processing (displays sentiment, does not yet display LIME explanations from API).

## Project Structure

```
.
├── AGENTS.md                 # Instructions for AI agent development
├── api/
│   ├── __init__.py
│   └── app.py                # FastAPI application
├── dashboard/
│   ├── __init__.py
│   └── app.py                # Streamlit dashboard
├── trained_models/           # Saved trained models and vectorizers
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Text preprocessing (spaCy based)
│   ├── models.py             # Model training (LogReg, SVM, RF), evaluation
│   ├── interpretability.py   # LIME explanation logic
│   └── utils.py              # Utility functions
├── tests/                    # Unit and integration tests (to be populated more)
├── requirements.txt          # Python dependencies
├── train_model.py            # Script to train sentiment models
├── README.md                 # This file
└── .gitignore                # Standard Python .gitignore (recommended to add)
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy English model:**
    The system uses spaCy for text preprocessing. Download the small English model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **(Optional) Download NLTK resources:**
    NLTK is not currently used for core processing but might be useful for other tasks or future features. If you encounter NLTK `LookupError` for any reason:
    ```bash
    python -m nltk.downloader stopwords wordnet omw-1.4 punkt
    ```

## Training Models

The `train_model.py` script is used to train sentiment analysis models. It uses the IMDb dataset (downloaded automatically via Hugging Face `datasets`).

**Usage:**
```bash
python train_model.py [OPTIONS]
```

**Key Options:**

*   `--model_type TEXT`: Type of model to train.
    Choices: `logistic_regression`, `svm`, `random_forest`.
    Default: `logistic_regression`.
*   `--max_features INTEGER`: Maximum number of features for TF-IDF. Default: `5000`.
*   `--data_subset INTEGER`: Number of samples from the original training data to use for a quick run (0 for full dataset). Useful for testing. Default: `0`.
*   `--tune_hyperparameters`: Enable hyperparameter tuning (GridSearchCV) for supported models (currently Logistic Regression).
*   `--model_file TEXT`: Filename for saving the trained model. Default: `logistic_regression_model.joblib` (will be adjusted based on `model_type`, e.g., `svm_model.joblib`).
*   `--vectorizer_file TEXT`: Filename for saving the TF-IDF vectorizer. Default: `tfidf_vectorizer.joblib`.

**Examples:**

*   Train a Logistic Regression model with default settings:
    ```bash
    python train_model.py
    ```
*   Train an SVM model using a subset of 2000 samples:
    ```bash
    python train_model.py --model_type svm --data_subset 2000
    ```
*   Train a Logistic Regression model with hyperparameter tuning on the full dataset:
    ```bash
    python train_model.py --model_type logistic_regression --tune_hyperparameters
    ```

Trained models and the TF-IDF vectorizer are saved to the `trained_models/` directory.

## Running the System

### 1. Run the FastAPI Backend

The API serves sentiment predictions and LIME explanations. Make sure you have trained a model first (e.g., `python train_model.py`).

The API will try to load `trained_models/logistic_regression_model.joblib` and `trained_models/tfidf_vectorizer.joblib` by default. If you trained a different model type (e.g., SVM), you'll need to update `MODEL_FILENAME` in `api/app.py` or ensure your desired model is named as the default.

Navigate to the project root and run:
```bash
python api/app.py
```
The API will be available at `http://localhost:8000`. Interactive documentation (Swagger UI) is at `http://localhost:8000/docs`.

### 2. Run the Streamlit Dashboard

The dashboard provides a UI for sentiment analysis. Ensure the FastAPI backend is running.

Navigate to the project root and run:
```bash
streamlit run dashboard/app.py
```
The dashboard will typically open at `http://localhost:8501`.

## API Usage

**Endpoint:** `POST /predict/`

**Request Body (JSON):**
```json
{
  "text": "Your text to analyze here"
}
```

**Response Body (JSON):**
```json
{
  "text": "Your text to analyze here",
  "sentiment": "positive", // or "negative"
  "confidence_score": 0.887, // Example, actual score from model
  "explanation": { // LIME explanation: words and their contribution scores
    "word1": 0.15,
    "word2": -0.08,
    // ... up to num_features specified in LIME (default 5 in current API)
  }
}
```
If LIME explanation fails for some reason, the `explanation` field might be `null`.

**Example with `curl`:**
```bash
curl -X POST "http://localhost:8000/predict/" \
-H "Content-Type: application/json" \
-d '{"text": "This movie is absolutely fantastic and a joy to watch!"}'
```

Expected output snippet:
```json
{
  "text": "This movie is absolutely fantastic and a joy to watch!",
  "sentiment": "positive",
  "confidence_score": 0.923, // value will vary
  "explanation": {
    "fantastic": 0.12,
    "joy": 0.10,
    // ...
  }
}
```

## Future Work & Advanced Features (Planned/Conceptual)

The following features are planned for future development:

*   **Sentiment Sensitivity Engine**: Adjust model sensitivity to sarcasm, specific emotions.
*   **Domain Adaptation**: Fine-tune or adapt models for different text domains (e.g., social media vs. product reviews).
*   **Multi-Language Support**: Extend beyond English using multilingual models.
*   **SHAP Interpretability**: Add SHAP as another explainability option.
*   **Active Learning Loop**: Allow model improvement through user feedback on uncertain predictions.
*   **Enhanced Dashboard**: Display LIME explanations visually.
*   **More Sophisticated Models**: Explore LSTM/GRU and Transformer-based models (e.g., BERT).
*   **Comprehensive Testing**: Expand unit and integration tests.
*   **Deployment**: Dockerization and cloud deployment instructions.

## Contributing
(Placeholder for contribution guidelines if this were an open project)

---
This README will be updated as the project evolves.
```
