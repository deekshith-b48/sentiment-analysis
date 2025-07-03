# api/app.py
"""
FastAPI application for the Sentiment Analysis service.

This API provides an endpoint for predicting sentiment of a given text and
returning LIME-based explanations for the prediction.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple

import joblib
import os
import sys
import numpy as np # For type hinting model predictions

# Add src directory to Python path to import modules from src
current_dir_path: str = os.path.dirname(os.path.abspath(__file__))
src_path: str = os.path.join(current_dir_path, '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from preprocessing import preprocess_text_spacy
from models import MODEL_DIR # Assuming SklearnModel type hint is in models.py or defined globally
from models import SklearnModel # Explicitly import if defined and used for type hint
from interpretability import explain_instance_lime, PreprocessorFunc

# --- Pydantic Models for Request and Response ---
class SentimentRequest(BaseModel):
    """Request model for sentiment prediction."""
    text: str = Field(..., example="This is a wonderful movie, I really enjoyed it!")
    language: str = Field("en", example="en", description="Language of the text (currently only 'en' is effectively supported).")

class SentimentResponse(BaseModel):
    """Response model for sentiment prediction."""
    text: str = Field(..., example="This is a wonderful movie, I really enjoyed it!")
    sentiment: str = Field(..., example="positive")
    confidence_score: Optional[float] = Field(None, example=0.95, description="Confidence score of the prediction, if available.")
    explanation: Optional[Dict[str, float]] = Field(None, example={"wonderful": 0.2, "movie": 0.1}, description="LIME explanation as word-weight pairs.")

# --- Application Setup ---
app: FastAPI = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment classification with model interpretability (LIME).",
    version="0.2.0" # Updated version
)

# --- Global Variables / Model Loading ---
# These will be loaded at startup and should be typed.
# For a generic sklearn model loaded by joblib, 'Any' or a specific base class can be used.
# If SklearnModel from models.py is a Union or specific type, use that.
loaded_model: Optional[SklearnModel] = None
loaded_vectorizer: Optional[Any] = None # Typically TfidfVectorizer, but joblib loads as 'Any'

# Default filenames for the model and vectorizer to be loaded.
# These should match the files saved by the training script (e.g., train_model.py).
# Consider making these configurable (e.g., via environment variables).
DEFAULT_MODEL_FILENAME: str = "logistic_regression_model.joblib"
DEFAULT_VECTORIZER_FILENAME: str = "tfidf_vectorizer.joblib"

# Sentiment labels mapping: model output index to human-readable label.
# Ensure this matches the labels used during training.
SENTIMENT_LABELS: Dict[int, str] = {0: "negative", 1: "positive"}


@app.on_event("startup")
async def load_resources() -> None:
    """
    Asynchronous event handler to load machine learning models and vectorizers at application startup.
    Populates global `loaded_model` and `loaded_vectorizer`.
    """
    global loaded_model, loaded_vectorizer # Allow modification of global variables

    model_filename: str = os.environ.get("MODEL_FILE", DEFAULT_MODEL_FILENAME)
    vectorizer_filename: str = os.environ.get("VECTORIZER_FILE", DEFAULT_VECTORIZER_FILENAME)

    model_path: str = os.path.join(MODEL_DIR, model_filename)
    vectorizer_path: str = os.path.join(MODEL_DIR, vectorizer_filename)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. API cannot serve predictions accurately.")
        print("Please ensure a model is trained and available at the specified path, or use dummy models for testing.")
        # Optionally, load dummy models here if strict startup is not required
        # For now, we allow startup but predictions will fail if model is None.
    else:
        try:
            loaded_model = joblib.load(model_path)
            print(f"Model '{model_filename}' loaded successfully from {model_path}.")
        except Exception as e:
            print(f"Error loading model '{model_filename}' from {model_path}: {e}")
            loaded_model = None # Ensure it's None if loading fails

    if not os.path.exists(vectorizer_path):
        print(f"ERROR: Vectorizer file not found at {vectorizer_path}. API cannot serve predictions accurately.")
    else:
        try:
            loaded_vectorizer = joblib.load(vectorizer_path)
            print(f"Vectorizer '{vectorizer_filename}' loaded successfully from {vectorizer_path}.")
        except Exception as e:
            print(f"Error loading vectorizer '{vectorizer_filename}' from {vectorizer_path}: {e}")
            loaded_vectorizer = None

    if loaded_model and loaded_vectorizer:
        print("API resources (model and vectorizer) loaded and ready.")
    else:
        print("Warning: Not all API resources could be loaded. Predictions may fail or be inaccurate.")
        # Consider creating dummy models here if you want the API to run for testing other endpoints
        # For example, see the __main__ block for dummy creation.

# --- API Endpoints ---
@app.get("/", tags=["General"], summary="Root endpoint for API health check and information.")
async def read_root() -> Dict[str, Any]:
    """
    Root endpoint providing basic API information and links to documentation.
    """
    return {
        "message": "Welcome to the Sentiment Analysis API!",
        "version": app.version,
        "status": "healthy" if loaded_model and loaded_vectorizer else "degraded (model/vectorizer not loaded)",
        "docs_url": app.docs_url,
        "redoc_url": app.redoc_url
    }

@app.post("/predict/", response_model=SentimentResponse, tags=["Sentiment Analysis"], summary="Predict sentiment of a text string.")
async def predict_sentiment(request: SentimentRequest) -> SentimentResponse:
    """
    Predicts the sentiment of a given text string and provides LIME-based explanations.

    - **text**: The input text string for sentiment analysis.
    - **language**: The language of the text (currently "en" is supported).

    Returns a `SentimentResponse` containing the original text, predicted sentiment,
    confidence score (if available), and LIME word importances.
    """
    global loaded_model, loaded_vectorizer # Access global loaded resources

    if loaded_model is None or loaded_vectorizer is None:
        raise HTTPException(status_code=503, detail="Model or vectorizer not loaded. API is not ready to make predictions.")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # 1. Preprocess the input text
        processed_text: str = preprocess_text_spacy(request.text)

        # 2. Vectorize the preprocessed text
        # Vectorizer expects a list of documents (even if it's just one)
        vectorized_text: Any = loaded_vectorizer.transform([processed_text])

        # 3. Make prediction
        prediction_array: np.ndarray = loaded_model.predict(vectorized_text)
        predicted_label_index: int = int(prediction_array[0]) # Get the first (and only) prediction

        # 4. Get confidence score (if available)
        confidence: Optional[float] = None
        if hasattr(loaded_model, "predict_proba"):
            probabilities: np.ndarray = loaded_model.predict_proba(vectorized_text)
            # Max probability for any class
            # confidence = float(np.max(probabilities[0]))
            # Probability of the *predicted* class:
            confidence = float(probabilities[0][predicted_label_index])


        # 5. Map prediction to a human-readable sentiment label
        sentiment: str = SENTIMENT_LABELS.get(predicted_label_index, "unknown_label")

        # 6. Add explanation from LIME
        explanation_list: Optional[List[Tuple[str, float]]] = None
        lime_explanation_dict: Optional[Dict[str, float]] = None
        try:
            # Determine class names for LIME based on the model's classes_ attribute
            if hasattr(loaded_model, 'classes_'):
                # Ensure SENTIMENT_LABELS covers all classes from the model
                lime_class_names: List[str] = [SENTIMENT_LABELS.get(int(cls_idx), f"Class {cls_idx}") for cls_idx in loaded_model.classes_]
            else:
                # Fallback if model.classes_ is not available (less common for sklearn classifiers)
                # This order must match the order of probabilities from predict_proba
                lime_class_names = [SENTIMENT_LABELS[i] for i in sorted(SENTIMENT_LABELS.keys())]


            explanation_list = explain_instance_lime(
                raw_text=request.text,
                model=loaded_model,
                vectorizer=loaded_vectorizer,
                preprocessor_func=preprocess_text_spacy, # Pass the actual preprocessor function
                class_names=lime_class_names,
                num_features=5 # Number of words to show in explanation
            )
            if explanation_list:
                lime_explanation_dict = dict(explanation_list)

        except Exception as lime_exc:
            print(f"LIME explanation generation failed: {lime_exc}")
            # lime_explanation_dict remains None

        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            confidence_score=confidence,
            explanation=lime_explanation_dict
        )

    except Exception as e:
        print(f"Error during prediction for text '{request.text[:50]}...': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# Placeholder Endpoints for Future Features (can be expanded later)
# @app.post("/adapt-domain/", tags=["Advanced Features"], summary="Adapt model to a new domain (Not Implemented).")
# async def adapt_domain(domain_data: Dict[Any, Any]):
#     raise HTTPException(status_code=501, detail="Domain adaptation feature not implemented yet.")

# @app.post("/feedback/", tags=["Advanced Features"], summary="Submit feedback for active learning (Not Implemented).")
# async def active_learning_feedback(feedback_data: Dict[Any, Any]):
#     raise HTTPException(status_code=501, detail="Active learning feedback feature not implemented yet.")


if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for local development (api/app.py)...")

    # Check if actual model/vectorizer files exist, otherwise create dummies for development
    # This allows the API to start even if `train_model.py` hasn't been run yet.
    # Dummy models/vectorizers will likely yield nonsensical results.
    model_file_to_check = os.path.join(MODEL_DIR, os.environ.get("MODEL_FILE", DEFAULT_MODEL_FILENAME))
    vectorizer_file_to_check = os.path.join(MODEL_DIR, os.environ.get("VECTORIZER_FILE", DEFAULT_VECTORIZER_FILENAME))

    if not os.path.exists(model_file_to_check):
        print(f"Warning: Model file '{model_file_to_check}' not found. Creating a dummy model.")
        from sklearn.linear_model import LogisticRegression
        dummy_model = LogisticRegression()
        # Fit with minimal data to make it "trained" and have classes_ attribute
        dummy_model.fit(np.array([[0],[1]]), np.array([0,1]))
        joblib.dump(dummy_model, model_file_to_check)

    if not os.path.exists(vectorizer_file_to_check):
        print(f"Warning: Vectorizer file '{vectorizer_file_to_check}' not found. Creating a dummy vectorizer.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        dummy_vectorizer = TfidfVectorizer()
        dummy_vectorizer.fit(["sample text for dummy vectorizer"]) # Fit with minimal data
        joblib.dump(dummy_vectorizer, vectorizer_file_to_check)

    # The load_resources function will be called by FastAPI on startup.
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # To run:  python api/app.py
    # Then access docs at http://localhost:8000/docs
    # Example POST request using curl:
    # curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json" -d '{"text": "This is a wonderful day!"}'
```
