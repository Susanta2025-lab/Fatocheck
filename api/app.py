from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import joblib
from pathlib import Path

# 1. Initialize the FastAPI application
app = FastAPI(
    title="Fatocheck - Fake News Detection API",
    description="A FastAPI service serving an optimized XGBoost NLP pipeline to classify news articles.",
    version="1.0.0"
)
# 2. Define the path to your serialized pipeline artifact
# Since app.py runs from the api/ folder, navigate up to the root, then into
MODEL_PATH = Path(__file__).resolve().parents[1] / "models/trained/xgboost_pipeline.joblib"
model_pipeline = joblib.load(MODEL_PATH)
# This loads the entire pipeline (TfidfVectorizer + XGBoost classifier) and caches it in memory on startup
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# 3. Define the Pydantic schema for incoming request validation
class NewsArticle(BaseModel):
    title: str | None = ""
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Trump’s allies are already lining up to apply to his $1.8 billion fund",
                "text": "Those who might claim they were wrongly targeted by the government include Jan. 6 rioters and people who tried to help Trump overturn the 2020 election."
            }
        }

# 4. Define API Endpoints
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

# 5. Define the prediction endpoint
@app.post("/predict")
def predict_news(article: NewsArticle):
    try:
        full_text = f"{article.title} {article.text}".strip()

        if not full_text:
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        prediction_code = int(model_pipeline.predict([full_text])[0])

        label_mapping = {
            0: "FAKE",
            1: "REAL"
        }

        label = label_mapping[prediction_code]

        proba = model_pipeline.predict_proba([full_text])[0]

        return {
            "prediction_code": prediction_code,
            "label": label,
            "probabilities": {
                "fake": float(proba[0]),
                "real": float(proba[1])
            },
            "model": "xgboost_pipeline"
        }
    except HTTPException as e:
        raise e # Re-raise HTTP exceptions to be handled by FastAPI
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
