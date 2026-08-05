# api/app.py

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware

from utils.inference import health_check, predict_news
from utils.request_context import request_id_ctx
from utils.settings import configure_logging, get_settings

# =========================================================
# Logging
# =========================================================

configure_logging()
logger = logging.getLogger(__name__)

APP_VERSION = "2.0.0"
REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time"


# =========================================================
# Lifespan
# =========================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — preserves previous startup behaviour."""

    status = health_check()

    logger.info("Fatocheck API starting")
    logger.info("System status: %s", status)

    if not status["xgboost_available"]:
        logger.warning("XGBoost model is unavailable")

    if not status["bert_artifact_available"]:
        logger.warning("Local BERT artifact is missing or incomplete")

    yield


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="FatoCheck - Fake News Detection API",
    description=(
        "Fake-news classification API using a production TF-IDF + XGBoost "
        "pipeline and optional locally fine-tuned BERT inference. "
        "Responses include X-Request-ID and X-Process-Time headers."
    ),
    version=APP_VERSION,
    contact={
        "name": "Susanta Hazra",
        "url": "https://github.com/Susanta2025-lab/Fatocheck",
    },
    openapi_tags=[
        {"name": "system", "description": "Service status and readiness"},
        {"name": "models", "description": "Model metadata and availability"},
        {"name": "prediction", "description": "Fake-news classification"},
    ],
    lifespan=lifespan,
)


# =========================================================
# Middleware
# =========================================================


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request ID, timing, and safe security headers."""

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming.strip() if incoming and incoming.strip() else str(uuid.uuid4())
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start
            logger.exception(
                "Unhandled error method=%s path=%s request_id=%s duration_s=%.4f",
                request.method,
                request.url.path,
                request_id,
                duration,
            )
            request_id_ctx.reset(token)
            raise

        duration = time.perf_counter() - start
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{duration:.4f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"

        logger.info(
            "request method=%s path=%s status=%s request_id=%s duration_s=%.4f",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
            duration,
        )
        request_id_ctx.reset(token)
        return response


app.add_middleware(RequestContextMiddleware)


# =========================================================
# Error helpers
# =========================================================


def _current_request_id(request: Request | None = None) -> str:
    if request is not None:
        return getattr(request.state, "request_id", None) or request_id_ctx.get()
    return request_id_ctx.get()


def _error_code_for_status(status_code: int) -> str:
    mapping = {
        400: "BAD_REQUEST",
        404: "NOT_FOUND",
        422: "VALIDATION_ERROR",
        503: "SERVICE_UNAVAILABLE",
        500: "INTERNAL_ERROR",
    }
    return mapping.get(status_code, "HTTP_ERROR")


def _error_payload(code: str, message: str, request_id: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        },
    }


def _http_exception_message(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        # Preserve useful validation-style details without dumping internals.
        return "; ".join(str(item) for item in detail)
    return str(detail)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = _current_request_id(request)
    message = _http_exception_message(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(_error_code_for_status(exc.status_code), message, request_id),
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = _current_request_id(request)
    errors = exc.errors()
    messages = []
    for err in errors:
        loc = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
        msg = err.get("msg", "Invalid value")
        messages.append(f"{loc}: {msg}" if loc else msg)
    message = "; ".join(messages) if messages else "Request validation failed"
    return JSONResponse(
        status_code=422,
        content=_error_payload("VALIDATION_ERROR", message, request_id),
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = _current_request_id(request)
    logger.exception(
        "Unhandled exception method=%s path=%s request_id=%s",
        request.method,
        request.url.path,
        request_id,
    )
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            "INTERNAL_ERROR",
            "Internal error while processing the request.",
            request_id,
        ),
        headers={REQUEST_ID_HEADER: request_id},
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
                "title": "Scientists announce a new AI system",
                "text": ("Researchers published details of a system for detecting misinformation."),
                "model": "xgboost",
            }
        }
    )


# =========================================================
# Root endpoint
# =========================================================


@app.get("/", tags=["system"])
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
        "version": APP_VERSION,
        "available_models": available_models,
        "xgboost_ready": status["xgboost_available"],
        "bert_artifact_available": status["bert_artifact_available"],
        "bert_loaded": status["bert_loaded"],
    }


# =========================================================
# Health and readiness
# =========================================================


@app.get("/health", tags=["system"])
def health():
    status = health_check()

    if not status["xgboost_available"]:
        raise HTTPException(
            status_code=503,
            detail="XGBoost production model is unavailable.",
        )

    return {
        "status": "healthy",
        "bert_artifact_available": status["bert_artifact_available"],
        "bert_loaded": status["bert_loaded"],
    }


@app.get("/ready", tags=["system"])
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


@app.get("/models", tags=["models"])
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
                "description": "TF-IDF + tuned XGBoost pipeline",
                "status": ("production" if status["xgboost_available"] else "unavailable"),
            },
            "bert": {
                "type": "Transformer",
                "description": "Locally fine-tuned BERT classifier",
                "status": bert_status,
            },
        }
    }


# =========================================================
# Prediction endpoint
# =========================================================


@app.post("/predict", tags=["prediction"])
def predict(article: NewsArticle):
    full_text = f"{article.title or ''}\n{article.text}".strip()
    request_id = request_id_ctx.get()

    if not full_text:
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty.",
        )

    try:
        logger.info(
            "prediction_start model=%s request_id=%s",
            article.model,
            request_id,
        )
        result = predict_news(
            text=full_text,
            model_type=article.model,
        )
        logger.info(
            "prediction_success model=%s prediction=%s request_id=%s",
            article.model,
            result.get("prediction"),
            request_id,
        )
        return {
            "success": True,
            "result": result,
        }

    except ValueError as exc:
        logger.warning(
            "prediction_rejected model=%s request_id=%s error_category=ValueError",
            article.model,
            request_id,
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except (FileNotFoundError, RuntimeError) as exc:
        # Log the real cause server-side; keep client message free of paths.
        logger.error(
            "prediction_unavailable model=%s request_id=%s error_category=%s detail=%s",
            article.model,
            request_id,
            type(exc).__name__,
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="The selected model is currently unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "prediction_failed model=%s request_id=%s error_category=unexpected",
            article.model,
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal error while running inference.",
        ) from exc
