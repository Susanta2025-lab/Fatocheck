## This FastAPI application (Latest Version) serves a combined machine learning model (XGBoost) and transformer model (BERT) for fake news detection.
# It loads a pre-trained XGBoost pipeline that includes text vectorization and classification.
# The API provides three endpoints: a health check at the root ("/"), a prediction endpoint ("/predict") and a optional model information endpoint ("/models").
# Second endpoint accepts news articles in JSON format and returns the predicted label along with probabilities.
#
# =========================================================
# Imports
# =========================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from utils.inference import predict_news


# =========================================================
# Initialize FastAPI Application
# =========================================================

app = FastAPI(
    title="Fatocheck-v2 - Fake News Detection API",

    description="""
    AI-powered fake news detection system using:

    - XGBoost NLP pipeline
    - BERT Transformer model

    Built with FastAPI.
    """,

    version="2.0.0"
)


# =========================================================
# Request Schema
# =========================================================

class NewsArticle(BaseModel):

    title: str | None = Field(
        default="",
        description="Optional news title"
    )

    text: str = Field(
        ...,
        description="Main news article text"
    )

    model: Literal["xgboost", "bert"] = Field(
        default="xgboost",
        description="Model to use for prediction"
    )

    class Config:

        json_schema_extra = {
            "example": {
                "title": "Scientists discover a revolutionary AI model",

                "text": """
                Researchers developed a new AI system capable
                of detecting misinformation with high accuracy.
                """,

                "model": "xgboost"
            }
        }


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def read_root():

    """
    Health check endpoint.
    """

    return {
        "status": "online",
        "project": "Fatocheck-v2: Fake News Detector-API-v2",
        "available_models": [
            "xgboost",
            "bert"
        ]
    }


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post("/predict")
def predict(article: NewsArticle):

    try:

        # Combine title + text
        full_text = f"""
        {article.title}

        {article.text}
        """.strip()

        # Validate input
        if not full_text:

            raise HTTPException(
                status_code=400,
                detail="Input text cannot be empty."
            )

        # Run unified inference pipeline
        result = predict_news(
            text=full_text,
            model_type=article.model
        )

        return {
            "success": True,
            "result": result
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )


# =========================================================
# Optional Model Information Endpoint
# =========================================================

@app.get("/models")
def get_models():

    """
    Returns available production models.
    """

    return {
        "models": {
            "xgboost": {
                "type": "Classical Machine Learning",
                "description": "Fast TF-IDF + XGBoost pipeline"
            },

            "bert": {
                "type": "Transformer",
                "description": "bert-base-uncased transformer model"
            }
        }
    }
