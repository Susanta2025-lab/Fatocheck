from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# 1. Initialize the FastAPI application
app = FastAPI(
    title="Fatocheck - Fake News Detection API",
    description="A FastAPI service serving an optimized XGBoost NLP pipeline to classify news articles.",
    version="1.0.0"
)

# 2. Define the path to your serialized pipeline artifact
# Since app.py runs from the api/ folder, navigate up to the root, then into models/trained/
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained/xgboost_pipeline.joblib"))

# 3. Load the model pipeline globally on startup so it's cached in memory
if os.path.exists(MODEL_PATH):
    # This loads the entire pipeline (TfidfVectorizer + XGBoost classifier)
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"--- Model pipeline successfully loaded from {MODEL_PATH} ---")
else:
    raise FileNotFoundError(f"Serialized model pipeline not found at: {MODEL_PATH}. Please run your training notebook first.")


# 4. Define the Pydantic schema for incoming request validation
class NewsArticle(BaseModel):
    title: str = ""
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Trump’s allies are already lining up to apply to his $1.8 billion fund",
                "text": "Those who might claim they were wrongly targeted by the government include Jan. 6 rioters and people who tried to help Trump overturn the 2020 election."
            }
        }


# 5. Define API Endpoints
@app.get("/")
def read_root():
    """
    Health check endpoint to verify the API is running and the model is loaded.
    """
    return {
        "status": "online",
        "project": "Fatocheck Fake News Detector",
        "model_loaded": "xgboost_pipeline"
    }


@app.post("/ predict")
def predict_news(article: NewsArticle):
    """
    Accepts a news article body (and optional title), combines them,
    runs them through the pipeline, and returns the classification.
    """
    try:
        # Combine title and text exactly the way you did during training preprocessing
        full_text = f"{article.title} {article.text}".strip()

        if not full_text:
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        # The pipeline expects an iterable (e.g., a list of strings)
        # It automatically performs .transform() via TF-IDF and then .predict()
        prediction_code = int(model_pipeline.predict([full_text])[0])

        # FINAL MAPPING:
        # Based on the processed dataset labels used during training
        label_mapping = {
            0: "Fake",
            1: "Real"
        }

        result_label = label_mapping.get(prediction_code, "Unknown")

        # Optional: If your pipeline supports predict_proba, get confidence scores
        if hasattr(model_pipeline, "predict_proba"):
            probabilities = model_pipeline.predict_proba([full_text])[0]
            confidence = float(probabilities[prediction_code])
        else:
            confidence = None

        return {
            "prediction_code": prediction_code,
            "label": result_label,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")
