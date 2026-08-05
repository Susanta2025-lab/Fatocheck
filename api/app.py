# api/app.py

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from utils.inference import health_check, predict_news
from utils.settings import configure_logging, get_settings


# =========================================================
# Logging
# =========================================================

configure_logging()
logger = logging.getLogger(__name__)


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Fatocheck - Fake News Detection API",
    description=(
        "Fake-news classification using a fast "
        "TF-IDF + XGBoost pipeline and an optional "
        "locally fine-tuned BERT model."
    ),
    version="2.0.0",
)


# =========================================================
# Request schema
# =========================================================

def _default_model_type() -> Literal["xgboost", "bert"]:
    return get_settings().default_model_type


class NewsArticle(BaseModel):
    title: str | None = Field(
        default="",
        description="Optional news headline",
    )

    text: str = Field(
        ...,
        min_length=1,
        description="News article text",
    )

    model: Literal["xgboost", "bert"] = Field(
        default_factory=_default_model_type,
        description="Inference model",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": (
                    "Scientists announce a new AI system"
                ),
                "text": (
                    "Researchers published details of a "
                    "system for detecting misinformation."
                ),
                "model": "xgboost",
            }
        }
    )


# =========================================================
# Startup
# =========================================================

@app.on_event("startup")
async def startup_event() -> None:
    status = health_check()

    logger.info("Fatocheck API starting")
    logger.info("System status: %s", status)

    if not status["xgboost_available"]:
        logger.warning("XGBoost model is unavailable")

    if not status["bert_artifact_available"]:
        logger.warning(
            "Local BERT artifact is missing or incomplete"
        )


# =========================================================
# Root endpoint
# =========================================================

@app.get("/")
def read_root():
    status = health_check()

    available_models = []

    if status["xgboost_available"]:
        available_models.append("xgboost")

    if status["bert_artifact_available"]:
        available_models.append("bert")

    return {
        "status": status["status"],
        "project": "Fatocheck - Fake News Detection API",
        "version": "2.0.0",
        "available_models": available_models,
        "xgboost_ready": status["xgboost_available"],
        "bert_artifact_available": (
            status["bert_artifact_available"]
        ),
        "bert_loaded": status["bert_loaded"],
    }


# =========================================================
# Health and readiness
# =========================================================

@app.get("/health")
def health():
    status = health_check()

    if not status["xgboost_available"]:
        raise HTTPException(
            status_code=503,
            detail="XGBoost production model is unavailable.",
        )

    return {
        "status": "healthy",
        "bert_artifact_available": (
            status["bert_artifact_available"]
        ),
        "bert_loaded": status["bert_loaded"],
    }


@app.get("/ready")
def readiness():
    status = health_check()

    if not status["xgboost_available"]:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready.",
        )

    return {"status": "ready"}


# =========================================================
# Models endpoint
# =========================================================

@app.get("/models")
def get_models():
    status = health_check()

    if status["bert_loaded"]:
        bert_status = "loaded"
    elif status["bert_artifact_available"]:
        bert_status = "lazy-loaded"
    else:
        bert_status = "unavailable"

    return {
        "models": {
            "xgboost": {
                "type": "Classical machine learning",
                "description": (
                    "TF-IDF + tuned XGBoost pipeline"
                ),
                "status": (
                    "production"
                    if status["xgboost_available"]
                    else "unavailable"
                ),
            },
            "bert": {
                "type": "Transformer",
                "description": (
                    "Locally fine-tuned BERT classifier"
                ),
                "status": bert_status,
            },
        }
    }


# =========================================================
# Prediction endpoint
# =========================================================

@app.post("/predict")
def predict(article: NewsArticle):
    full_text = (
        f"{article.title or ''}\n{article.text}"
    ).strip()

    if not full_text:
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty.",
        )

    try:
        result = predict_news(
            text=full_text,
            model_type=article.model,
        )

        return {
            "success": True,
            "result": result,
        }

    except ValueError as exc:
        logger.warning("Invalid prediction request: %s", exc)

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Model unavailable: %s", exc)

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected inference failure")

        raise HTTPException(
            status_code=500,
            detail="Internal error while running inference.",
        ) from exc
