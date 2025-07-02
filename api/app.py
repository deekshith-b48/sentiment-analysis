# api/app.py
"""
FastAPI application for the Sentiment Analysis service.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import sys

# Add src directory to Python path to import modules from src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from preprocessing import preprocess_text_spacy # For preprocessing
from models import MODEL_DIR # To locate models
from interpretability import explain_instance_lime # For LIME explanations
# SENTIMENT_LABELS is defined locally in this file

# --- Pydantic Models for Request and Response ---
class SentimentRequest(BaseModel):
    text: str
    language: str = "en" # Placeholder for multi-language support

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence_score: float = None # Optional: if model provides it
    explanation: dict = None # Optional: for LIME/SHAP explanations

# --- Application Setup ---
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment classification with advanced features.",
    version="0.1.0"
)

# --- Global Variables / Model Loading ---
# These will be loaded at startup
model = None
vectorizer = None
MODEL_FILENAME = "logistic_regression_model.joblib" # Default model
VECTORIZER_FILENAME = "tfidf_vectorizer.joblib" # Default vectorizer

# Define sentiment labels (adjust based on your model's output)
# Example: 0 -> negative, 1 -> neutral, 2 -> positive
# Or for binary: 0 -> negative, 1 -> positive
SENTIMENT_LABELS = {0: "negative", 1: "positive"} # Adjust if using neutral class


@app.on_event("startup")
async def load_resources():
    """
    Load models and other resources at application startup.
    """
    global model, vectorizer

    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    vectorizer_path = os.path.join(MODEL_DIR, VECTORIZER_FILENAME)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. API cannot serve predictions.")
        # In a real app, you might raise an error or prevent startup
        return
    if not os.path.exists(vectorizer_path):
        print(f"ERROR: Vectorizer file not found at {vectorizer_path}. API cannot serve predictions.")
        return

    try:
        model = joblib.load(model_path)
        print(f"Model '{MODEL_FILENAME}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{MODEL_FILENAME}': {e}")
        # Handle error appropriately

    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer '{VECTORIZER_FILENAME}' loaded successfully.")
    except Exception as e:
        print(f"Error loading vectorizer '{VECTORIZER_FILENAME}': {e}")
        # Handle error appropriately

    # Initialize other resources if needed (e.g., LIME explainer, emotion model)
    print("API resources loaded.")


# --- API Endpoints ---
@app.get("/", tags=["General"])
async def read_root():
    """
    Root endpoint providing basic API information.
    """
    return {
        "message": "Welcome to the Sentiment Analysis API!",
        "version": app.version,
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.post("/predict/", response_model=SentimentResponse, tags=["Sentiment Analysis"])
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for a given text.

    - **text**: The input text string for sentiment analysis.
    - **language**: The language of the text (default: "en"). Currently, only English is robustly supported by the base model.
    """
    global model, vectorizer

    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model or vectorizer not loaded. API is not ready.")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # 1. Preprocess the input text
        #    Ensure preprocessing matches what was used during training
        processed_text = preprocess_text_spacy(request.text)

        # 2. Vectorize the preprocessed text
        #    The vectorizer expects a list of documents
        vectorized_text = vectorizer.transform([processed_text])

        # 3. Make prediction
        prediction = model.predict(vectorized_text)
        predicted_label_index = prediction[0] # Get the first (and only) prediction

        # 4. Get confidence score (if available)
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vectorized_text)
            confidence = float(max(probabilities[0])) # Max probability for the predicted class
            # To get probability of the *predicted* class:
            # confidence = float(probabilities[0][predicted_label_index])


        # 5. Map prediction to a human-readable sentiment label
        sentiment = SENTIMENT_LABELS.get(predicted_label_index, "unknown")

        # 6. Add explanation from LIME
        explanation_output = None
        try:
            # Ensure class_names are in the correct order for LIME (e.g., as per model.classes_)
            # For binary classification with labels 0 and 1, and SENTIMENT_LABELS = {0: 'neg', 1: 'pos'}
            # class_names for LIME should be ['negative', 'positive'] if that's how model was trained/outputs.
            # Let's derive class_names from SENTIMENT_LABELS keys if possible or define explicitly.
            # The order matters: it should match the order of probabilities from model.predict_proba()

            # Assuming model.classes_ gives the order [0, 1] (negative, positive)
            # and SENTIMENT_LABELS maps these indices to names.
            # lime_class_names = [SENTIMENT_LABELS[i] for i in sorted(SENTIMENT_LABELS.keys())]
            # More robustly, if model.classes_ is available:
            if hasattr(model, 'classes_'):
                lime_class_names = [SENTIMENT_LABELS.get(cls_idx, str(cls_idx)) for cls_idx in model.classes_]
            else: # Fallback if model.classes_ is not available (less common for sklearn)
                lime_class_names = [SENTIMENT_LABELS[0], SENTIMENT_LABELS[1]] # Assuming binary 0, 1

            if vectorizer and model: # Ensure they are loaded
                explanation_output = explain_instance_lime(
                    raw_text=request.text,
                    model=model,
                    vectorizer=vectorizer,
                    preprocessor_func=preprocess_text_spacy, # Pass the actual preprocessor
                    class_names=lime_class_names,
                    num_features=5 # Get top 5 features for the explanation
                )
        except Exception as lime_exc:
            print(f"LIME explanation failed: {lime_exc}")
            # explanation_output will remain None or you can set a specific error message

        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            confidence_score=confidence,
            explanation=dict(explanation_output) if explanation_output else None # Convert list of tuples to dict for Pydantic/JSON
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# --- Placeholder Endpoints for Future Features ---
# @app.post("/adapt-domain/", tags=["Advanced Features"])
# async def adapt_domain(domain_data: dict): # Define Pydantic model for domain_data
#     # Placeholder for domain adaptation logic
#     return {"message": "Domain adaptation endpoint (Not Implemented Yet)"}

# @app.post("/feedback/", tags=["Advanced Features"])
# async def active_learning_feedback(feedback_data: dict): # Define Pydantic model for feedback
#     # Placeholder for active learning feedback
#     return {"message": "Active learning feedback endpoint (Not Implemented Yet)"}


if __name__ == "__main__":
    import uvicorn
    # This is for local development.
    # For production, use a proper ASGI server like Uvicorn managed by Gunicorn.
    print("Starting Uvicorn server for local development...")
    print("Models and vectorizers need to be in the 'trained_models' directory and match filenames.")
    print(f"Expected model: {os.path.join(MODEL_DIR, MODEL_FILENAME)}")
    print(f"Expected vectorizer: {os.path.join(MODEL_DIR, VECTORIZER_FILENAME)}")

    # Create dummy model and vectorizer for testing if they don't exist
    # This is just to allow the API to start without actual trained files
    # In a real scenario, these files must be generated by the training script
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(MODEL_DIR, MODEL_FILENAME)):
        print(f"Warning: Dummy model created for {MODEL_FILENAME} as it was not found.")
        # Create a dummy sklearn model
        from sklearn.linear_model import LogisticRegression
        dummy_model = LogisticRegression()
        # Fit with minimal data to make it "trained"
        dummy_model.fit([[0]], [0])
        joblib.dump(dummy_model, os.path.join(MODEL_DIR, MODEL_FILENAME))

    if not os.path.exists(os.path.join(MODEL_DIR, VECTORIZER_FILENAME)):
        print(f"Warning: Dummy vectorizer created for {VECTORIZER_FILENAME} as it was not found.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        dummy_vectorizer = TfidfVectorizer()
        # Fit with minimal data
        dummy_vectorizer.fit(["sample text"])
        joblib.dump(dummy_vectorizer, os.path.join(MODEL_DIR, VECTORIZER_FILENAME))

    uvicorn.run(app, host="0.0.0.0", port=8000)

    # To run:  python api/app.py
    # Then access docs at http://localhost:8000/docs
    # Example POST request using curl:
    # curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json" -d '{"text": "This is a wonderful day!"}'
    # curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json" -d '{"text": "I am feeling very sad and frustrated."}'
