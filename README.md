# Advanced Sentiment Analysis System

This project is an end-to-end Machine Learning-based Sentiment Analysis system. It aims to classify text into positive, negative, or neutral sentiment and includes several advanced features.

**Work in Progress - This README will be updated as features are implemented.**

## Features (Planned)

*   **Core Sentiment Classification**: Classifies text into positive, negative, (and potentially neutral) sentiment.
*   **Customizable Sentiment Sensitivity Engine**: Allows users to adjust model sensitivity to specific emotions (e.g., sarcasm, joy). (Future)
*   **Domain Adaptation Layer**: Enables adaptation to different domains without full retraining. (Future)
*   **Interactive Sentiment Trend Visualization Dashboard**: Visualizes sentiment changes using Streamlit.
*   **Model Interpretability with LIME/SHAP**: Explains predictions by highlighting key words. (Future)
*   **Active Learning Loop**: Improves the model over time with user feedback. (Future)
*   **Multi-Language Support**: Aims to classify sentiment in multiple languages. (Future)
*   **RESTful API**: Deploys the model as a real-time service using FastAPI.

## Project Structure

```
.
├── AGENTS.md                 # Instructions for AI agent development
├── api/                      # FastAPI application
│   ├── __init__.py
│   └── app.py                # Main API logic
├── dashboard/                # Streamlit dashboard application
│   ├── __init__.py
│   └── app.py                # Main dashboard logic
├── data/                     # Raw and processed datasets (will be populated)
├── trained_models/           # Saved trained models and vectorizers (will be populated by training script)
├── notebooks/                # Jupyter notebooks for experimentation and exploration (optional)
├── src/                      # Source code for core logic
│   ├── __init__.py
│   ├── preprocessing.py      # Text preprocessing functions
│   ├── models.py             # Model training, loading, evaluation
│   └── utils.py              # Utility functions, data loaders
├── tests/                    # Unit and integration tests (will be populated)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Dockerfile                # For containerizing the application (Future)
└── train_model.py            # Script to train models (To be created)
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK resources:**
    It is **highly recommended** to download the necessary NLTK resources manually before running the application for the first time. Open a Python interpreter or run the following command in your terminal:
    ```bash
    python -m nltk.downloader stopwords wordnet omw-1.4 punkt
    ```
    The application will attempt to download them if missing, but manual download is more reliable.

5.  **Download spaCy model (if you plan to use spaCy preprocessing features in the future):**
    The current `preprocessing.py` has spaCy functions commented out. If you enable them, you'll need the model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Running the System

The system consists of two main parts: the **FastAPI backend** and the **Streamlit dashboard**. You'll typically need to run the backend first.

### 1. Training a Model (Placeholder - Script to be created)

Before running the API or Dashboard, you need a trained model and its corresponding vectorizer.
A `train_model.py` script will be provided. For now, the `api/app.py` can create dummy models if real ones are not found, allowing the API to start for basic testing.

The `train_model.py` script (once created) will:
*   Load data (e.g., from the `data/` directory).
*   Preprocess the text.
*   Train a sentiment analysis model (e.g., Logistic Regression).
*   Save the trained model (e.g., `logistic_regression_model.joblib`) and the TF-IDF vectorizer (e.g., `tfidf_vectorizer.joblib`) to the `trained_models/` directory.

**Example (Conceptual):**
```bash
python train_model.py --data_path data/your_dataset.csv --text_column review_text --label_column sentiment
```

### 2. Running the FastAPI Backend

The API serves sentiment predictions.
Navigate to the project root directory.

```bash
python api/app.py
```

This will start the Uvicorn server, typically on `http://localhost:8000`.
The API documentation (Swagger UI) will be available at `http://localhost:8000/docs`.

**Note:** The API expects trained model files (`logistic_regression_model.joblib` and `tfidf_vectorizer.joblib` by default) to be present in the `trained_models/` directory. If these are not found, the API (as currently written in `api/app.py`) will create and use dummy versions for basic functionality, printing a warning.

### 3. Running the Streamlit Dashboard

The dashboard provides a user interface to interact with the sentiment analysis API.
Navigate to the project root directory.

```bash
streamlit run dashboard/app.py
```

This will typically open the dashboard in your web browser at `http://localhost:8501`.

## Usage

### API Endpoints

*   `POST /predict/`:
    *   Accepts JSON input: `{"text": "Your text here"}`
    *   Returns JSON output: `{"text": "Your text here", "sentiment": "positive/negative/neutral", "confidence_score": 0.95, "explanation": null}` (explanation to be added later)

**Example with `curl`:**
```bash
curl -X POST "http://localhost:8000/predict/" \
-H "Content-Type: application/json" \
-d '{"text": "This is a wonderful movie, I really enjoyed it!"}'
```

### Dashboard
*   **Single Text Analysis**: Enter text into the text area and click "Analyze Sentiment".
*   **Batch Analysis (CSV)**: Upload a CSV file. Select the column containing text data and click "Analyze Batch". Results and visualizations will be displayed.

## Unique Features (Planned Implementation Details)

*   **Customizable Sentiment Sensitivity Engine**:
    *   *How it will work (concept)*: Integrate an emotion detection model. Assign weights to different detected emotions. Users can adjust these weights to make the overall sentiment prediction more or less sensitive to certain underlying emotions. For example, increasing the weight for "sarcasm" (if detectable) might flip a seemingly positive statement to negative.
*   **Domain Adaptation Layer**:
    *   *How it will work (concept)*: Utilize transfer learning. Fine-tune a base pre-trained model (e.g., BERT) on smaller, domain-specific datasets. Alternatively, explore using adapter modules which are small, learnable layers inserted into a pre-trained model.
*   **Interactive Sentiment Trend Visualization Dashboard**:
    *   *Implementation*: Using Streamlit and Plotly for charts. Will display sentiment distributions, and potentially trends if time-series data is available/simulated.
*   **Model Interpretability with LIME/SHAP**:
    *   *How it will work (concept)*: Integrate LIME (Local Interpretable Model-agnostic Explanations) or SHAP (SHapley Additive exPlanations) to identify which words in the input text contributed most to the sentiment prediction.
*   **Active Learning Loop**:
    *   *How it will work (concept)*: Identify predictions where the model is uncertain (e.g., low confidence scores). Present these to a user for manual labeling. Periodically retrain the model incorporating this new feedback.
*   **Multi-Language Support**:
    *   *How it will work (concept)*: Use multilingual transformer models (e.g., XLM-RoBERTa) that are pre-trained on many languages. Fine-tune on multilingual sentiment datasets.

## Further Improvements and Extensions (Suggestions)

*   More sophisticated preprocessing (e.g., handling negations, emojis).
*   Advanced embedding techniques (BERT, RoBERTa, etc.).
*   Hyperparameter tuning for all models.
*   More robust error handling and logging.
*   Scalable deployment architecture (e.g., Kubernetes).
*   User authentication for API and dashboard.
*   Database integration for storing feedback from active learning or user data.
*   Real-time processing of streaming data (e.g., social media feeds).

---

This README provides an initial outline. It will be updated with more specific instructions and details as the project progresses.
```
