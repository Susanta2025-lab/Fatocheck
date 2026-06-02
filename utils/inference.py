# utils/inference.py

import time
import joblib
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)

# =========================================================
# Base Directory
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models" / "trained"

# =========================================================
# Initialize Preprocessor
# =========================================================

processor = TextPreprocessor()

# =========================================================
# Model Paths
# =========================================================

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_pipeline.joblib"
BERT_MODEL_NAME = "bert-base-uncased"
BERT_CACHE_DIR = MODELS_DIR / "bert-cache"

# =========================================================
# Load XGBoost Model (Immediately)
# =========================================================

xgboost_model = None
try:
    xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
    logger.info("✅ XGBoost model loaded successfully")
except FileNotFoundError:
    logger.warning(f"⚠️  XGBoost model not found at {XGBOOST_MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Error loading XGBoost: {e}")

# =========================================================
# Lazy Load BERT (On First Use)
# =========================================================

_tokenizer = None
_bert_model = None
_bert_loading = False

def load_bert_model():
    """Lazy load BERT model on first request"""
    global _tokenizer, _bert_model, _bert_loading

    if _tokenizer is not None and _bert_model is not None:
        return _tokenizer, _bert_model

    if _bert_loading:
        logger.warning("⏳ BERT model is already loading, please wait...")
        # Wait for loading to complete
        import time
        time.sleep(2)
        if _tokenizer is not None and _bert_model is not None:
            return _tokenizer, _bert_model

    _bert_loading = True
    logger.info("📥 Loading BERT model... (first time, may take a minute)")

    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            BERT_MODEL_NAME,
            cache_dir=str(BERT_CACHE_DIR)
        )
        _bert_model = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME,
            cache_dir=str(BERT_CACHE_DIR)
        )
        _bert_model.eval()
        logger.info("✅ BERT model loaded successfully")
        _bert_loading = False
        return _tokenizer, _bert_model
    except Exception as e:
        _bert_loading = False
        logger.error(f"❌ Failed to load BERT: {e}")
        raise

# =========================================================
# Label Mapping
# =========================================================

LABEL_MAP = {
    0: "Fake",
    1: "Real"
}

# =========================================================
# XGBoost Prediction
# =========================================================

def predict_xgboost(text):
    """Fast XGBoost prediction"""
    if xgboost_model is None:
        raise RuntimeError("XGBoost model not available")

    start_time = time.time()

    try:
        cleaned_text = processor.clean_text(text)
        prediction = xgboost_model.predict([cleaned_text])[0]
        probabilities = xgboost_model.predict_proba([cleaned_text])[0]
        confidence = round(float(max(probabilities)), 4)

        return {
            "model": "xgboost",
            "prediction": LABEL_MAP[prediction],
            "confidence": confidence,
            "processing_time_seconds": round(time.time() - start_time, 4)
        }
    except Exception as e:
        logger.error(f"XGBoost prediction error: {e}")
        raise

# =========================================================
# BERT Prediction
# =========================================================

def predict_bert(text):
    """BERT prediction with lazy loading"""
    start_time = time.time()

    try:
        # Lazy load model
        tokenizer, bert_model = load_bert_model()

        cleaned_text = processor.clean_text(text)
        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = bert_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        return {
            "model": "bert-base-uncased",
            "prediction": LABEL_MAP[prediction.item()],
            "confidence": round(confidence.item(), 4),
            "processing_time_seconds": round(time.time() - start_time, 4)
        }
    except Exception as e:
        logger.error(f"BERT prediction error: {e}")
        raise

# =========================================================
# Unified Prediction Router
# =========================================================

def predict_news(text, model_type="xgboost"):
    """Route prediction to appropriate model"""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    if model_type == "xgboost":
        return predict_xgboost(text)
    elif model_type == "bert":
        return predict_bert(text)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported: xgboost, bert"
        )

# =========================================================
# Health Check (No Model Loading)
# =========================================================

def health_check():
    """Quick health check without loading models"""
    return {
        "status": "healthy",
        "xgboost_available": xgboost_model is not None,
        "bert_available": _bert_model is not None
    }
