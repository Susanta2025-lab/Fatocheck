# utils/inference.py

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from utils.preprocessing import TextPreprocessor
from utils.settings import BERT_MAX_LENGTH, get_settings

logger = logging.getLogger(__name__)


# =========================================================
# Paths and configuration
# =========================================================
# Module-level aliases are retained so tests can monkeypatch them and so
# existing import sites keep working. Values come from the central Settings.

_settings = get_settings()
BASE_DIR = _settings.base_dir
MODELS_DIR = _settings.models_dir

XGBOOST_MODEL_PATH = _settings.xgboost_model_path
BERT_MODEL_PATH = _settings.bert_model_path

_BERT_REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)

_BERT_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
)


# =========================================================
# Preprocessor and label mappings
# =========================================================

processor = TextPreprocessor()

XGBOOST_LABEL_MAP: Dict[int, str] = {
    0: "Fake",
    1: "Real",
}

BERT_FALLBACK_LABEL_MAP: Dict[int, str] = {
    0: "Fake",
    1: "Real",
}


# =========================================================
# Load XGBoost immediately
# =========================================================

xgboost_model = None

try:
    xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
    logger.info("XGBoost model loaded successfully")

except FileNotFoundError:
    logger.warning(
        "XGBoost model not found at %s",
        XGBOOST_MODEL_PATH,
    )

except Exception:
    logger.exception("Failed to load XGBoost model")


# =========================================================
# Lazy BERT loading
# =========================================================

_tokenizer: Optional[PreTrainedTokenizerBase] = None
_bert_model: Optional[PreTrainedModel] = None
_bert_load_error: Optional[str] = None
_bert_lock = threading.Lock()


def _find_bert_weight_file() -> Optional[Path]:
    """Return the first supported BERT weight file found."""

    for filename in _BERT_WEIGHT_FILES:
        candidate = BERT_MODEL_PATH / filename

        if candidate.exists():
            return candidate

    return None


def is_bert_artifact_available() -> bool:
    """Check whether the local BERT artifact appears complete."""

    if not BERT_MODEL_PATH.is_dir():
        return False

    required_files_exist = all((BERT_MODEL_PATH / filename).is_file() for filename in _BERT_REQUIRED_FILES)

    return required_files_exist and _find_bert_weight_file() is not None


def _validate_bert_artifact() -> None:
    """Raise a clear error when local BERT files are incomplete."""

    if not BERT_MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Local BERT directory not found at {BERT_MODEL_PATH}.")

    missing_files = [filename for filename in _BERT_REQUIRED_FILES if not (BERT_MODEL_PATH / filename).is_file()]

    if missing_files:
        raise FileNotFoundError("Missing required BERT files: " f"{', '.join(missing_files)}.")

    if _find_bert_weight_file() is None:
        expected = ", ".join(_BERT_WEIGHT_FILES)

        raise FileNotFoundError(f"No BERT weight file found in {BERT_MODEL_PATH}. " f"Expected one of: {expected}.")


def load_bert_model() -> Tuple[
    PreTrainedTokenizerBase,
    PreTrainedModel,
]:
    """Load the locally fine-tuned BERT model on first use."""

    global _tokenizer, _bert_model, _bert_load_error

    if _tokenizer is not None and _bert_model is not None:
        return _tokenizer, _bert_model

    with _bert_lock:
        if _tokenizer is not None and _bert_model is not None:
            return _tokenizer, _bert_model

        try:
            _validate_bert_artifact()

            logger.info(
                "Loading local BERT model from %s",
                BERT_MODEL_PATH,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                BERT_MODEL_PATH,
                local_files_only=True,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                BERT_MODEL_PATH,
                local_files_only=True,
            )

            num_labels = getattr(model.config, "num_labels", None)

            if num_labels != 2:
                raise RuntimeError("Expected a binary BERT classifier, " f"but num_labels={num_labels}.")

            model.to("cpu")
            model.eval()

            _tokenizer = tokenizer
            _bert_model = model
            _bert_load_error = None

            logger.info("BERT model loaded successfully")

            return _tokenizer, _bert_model

        except FileNotFoundError as exc:
            _tokenizer = None
            _bert_model = None
            _bert_load_error = str(exc)
            logger.error("BERT artifact validation failed: %s", exc)
            raise

        except Exception as exc:
            _tokenizer = None
            _bert_model = None
            _bert_load_error = str(exc)

            logger.exception("Failed to load local BERT model")

            raise RuntimeError("The local BERT model could not be loaded.") from exc


def _resolve_bert_label(
    prediction_index: int,
    model: PreTrainedModel,
) -> str:
    """Resolve the model output to Fake or Real."""

    id2label = getattr(model.config, "id2label", None) or {}

    raw_label = id2label.get(
        prediction_index,
        id2label.get(str(prediction_index)),
    )

    if raw_label:
        normalized = str(raw_label).strip().lower()

        if not normalized.startswith("label_"):
            if normalized.startswith("fake"):
                return "Fake"

            if normalized.startswith("real"):
                return "Real"

    return BERT_FALLBACK_LABEL_MAP[prediction_index]


# =========================================================
# XGBoost inference
# =========================================================


def predict_xgboost(text: str) -> Dict[str, Any]:
    """Run prediction with the XGBoost pipeline."""

    if xgboost_model is None:
        raise RuntimeError("XGBoost model is unavailable. " f"Expected artifact at {XGBOOST_MODEL_PATH}.")

    start_time = time.perf_counter()

    cleaned_text = processor.clean_text(text)

    prediction = int(xgboost_model.predict([cleaned_text])[0])

    probabilities = xgboost_model.predict_proba([cleaned_text])[0]

    model_classes = list(xgboost_model.classes_)

    if prediction not in model_classes:
        raise RuntimeError(f"Predicted class {prediction} is missing " f"from model classes {model_classes}.")

    predicted_position = model_classes.index(prediction)
    confidence = float(probabilities[predicted_position])

    return {
        "model": "xgboost",
        "prediction": XGBOOST_LABEL_MAP[prediction],
        "confidence": round(confidence, 4),
        "processing_time_seconds": round(
            time.perf_counter() - start_time,
            4,
        ),
    }


# =========================================================
# BERT inference
# =========================================================


def predict_bert(text: str) -> Dict[str, Any]:
    """Run prediction with the local fine-tuned BERT model."""

    start_time = time.perf_counter()

    tokenizer, bert_model = load_bert_model()

    # Keep this preprocessing because the model was trained
    # using the clean_content column.
    cleaned_text = processor.clean_text(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=BERT_MAX_LENGTH,
    )

    with torch.inference_mode():
        outputs = bert_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    prediction_index = int(prediction.item())

    return {
        "model": "fatocheck-bert",
        "prediction": _resolve_bert_label(
            prediction_index,
            bert_model,
        ),
        "confidence": round(float(confidence.item()), 4),
        "processing_time_seconds": round(
            time.perf_counter() - start_time,
            4,
        ),
    }


# =========================================================
# Unified router
# =========================================================


def predict_news(
    text: str,
    model_type: str = "xgboost",
) -> Dict[str, Any]:
    """Route a request to the selected production model."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    if model_type == "xgboost":
        return predict_xgboost(text)

    if model_type == "bert":
        return predict_bert(text)

    raise ValueError(f"Unsupported model type: {model_type}. " "Supported models: xgboost, bert.")


# =========================================================
# Health information
# =========================================================


def health_check() -> Dict[str, Any]:
    """Return status without triggering BERT loading."""

    xgboost_available = xgboost_model is not None

    return {
        "status": ("healthy" if xgboost_available else "degraded"),
        "xgboost_available": xgboost_available,
        "bert_artifact_available": (is_bert_artifact_available()),
        "bert_loaded": _bert_model is not None,
        "bert_load_error": _bert_load_error,
    }
