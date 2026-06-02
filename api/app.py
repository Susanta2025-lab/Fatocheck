## FastAPI application for Fake News Detection
# =========================================================
# Imports
# =========================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import logging

from utils.inference import predict_news, health_check

# =========================================================
# Setup Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# Initialize FastAPI Application
# =========================================================

app = FastAPI(
    title="Fatocheck - Fake News Detection API",
    description="""
    AI-powered fake news detection system using:
    - XGBoost NLP pipeline (fast)
    - BERT Transformer model (accurate)

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
                "text": "Researchers developed a new AI system capable of detecting misinformation with high accuracy.",
                "model": "xgboost"
            }
        }

# =========================================================
# Startup Event - Log Initialization
# =========================================================

@app.on_event("startup")
async def startup_event():
    """Called when app starts - quick check"""
    logger.info("🚀 Fatocheck API starting...")
    status = health_check()
    logger.info(f"📊 System status: {status}")
    if not status["xgboost_available"]:
        logger.warning("⚠️  XGBoost model not available!")

# =========================================================
# Health Check Endpoint (Fast - Renders uses this)
# =========================================================

@app.get("/health")
def health():
    """Health check endpoint - Render uses this"""
    status = health_check()
    if not status["xgboost_available"]:
        return {"status": "degraded", "message": "XGBoost not available"}
    return {"status": "healthy"}

# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def read_root():
    """Health check endpoint"""
    status = health_check()
    return {
        "status": "online",
        "project": "Fatocheck - Fake News Detection API",
        "version": "2.0.0",
        "available_models": ["xgboost", "bert"],
        "xgboost_ready": status["xgboost_available"],
        "bert_ready": status["bert_available"]
    }

# =========================================================
# Prediction Endpoint
# =========================================================

@app.post("/predict")
def predict(article: NewsArticle):
    """Predict fake news from article text"""
    try:
        # Combine title + text
        full_text = f"{article.title}\n{article.text}".strip()

        # Validate input
        if not full_text:
            raise HTTPException(
                status_code=400,
                detail="Input text cannot be empty."
            )

        # Run prediction
        result = predict_news(
            text=full_text,
            model_type=article.model
        )

        return {
            "success": True,
            "result": result
        }

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )

# =========================================================
# Models Information Endpoint
# =========================================================

@app.get("/models")
def get_models():
    """Returns available production models"""
    return {
        "models": {
            "xgboost": {
                "type": "Classical Machine Learning",
                "description": "Fast TF-IDF + XGBoost pipeline",
                "inference_time": "~50-100ms",
                "status": "production"
            },
            "bert": {
                "type": "Transformer",
                "description": "bert-base-uncased transformer model",
                "inference_time": "~500-1000ms (first use slower)",
                "status": "lazy-loaded"
            }
        }
    }

# =========================================================
# Readiness Endpoint (For Kubernetes/Render)
# =========================================================

@app.get("/ready")
def readiness():
    """Readiness check - returns 200 if app is ready"""
    status = health_check()
    if not status["xgboost_available"]:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}
