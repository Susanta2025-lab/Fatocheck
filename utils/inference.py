# utils/inference.py


# =========================================================
# Imports
# =========================================================

import time
import joblib
import torch

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from utils.preprocessing import TextPreprocessor


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

XGBOOST_MODEL_PATH = (
    MODELS_DIR / "xgboost_pipeline.joblib"
)

BERT_MODEL_PATH = (
    MODELS_DIR / "bert-base-uncased"
)


# =========================================================
# Load XGBoost Model
# =========================================================

xgboost_model = joblib.load(XGBOOST_MODEL_PATH)


# =========================================================
# Load BERT Tokenizer + Model
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    BERT_MODEL_PATH
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL_PATH
)

bert_model.eval()


# =========================================================
# Label Mapping
# =========================================================

LABEL_MAP = {
    0: "Fake",
    1: "Real"
}


# =========================================================
# XGBoost Prediction Function
# =========================================================

def predict_xgboost(text):

    start_time = time.time()

    # Clean text
    cleaned_text = processor.clean_text(text)

    # Prediction
    prediction = xgboost_model.predict([cleaned_text])[0]

    # Confidence score
    probabilities = xgboost_model.predict_proba(
        [cleaned_text]
    )[0]

    confidence = round(
        float(max(probabilities)),
        4
    )

    end_time = time.time()

    return {
        "model": "xgboost",
        "prediction": LABEL_MAP[prediction],
        "confidence": confidence,
        "processing_time_seconds": round(
            end_time - start_time,
            4
        )
    }


# =========================================================
# BERT Prediction Function
# =========================================================

def predict_bert(text):

    start_time = time.time()

    # Clean text
    cleaned_text = processor.clean_text(text)

    # Tokenize input
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Disable gradient calculations
    with torch.no_grad():

        outputs = bert_model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    end_time = time.time()

    return {
        "model": "bert-base-uncased",
        "prediction": LABEL_MAP[
            prediction.item()
        ],
        "confidence": round(
            confidence.item(),
            4
        ),
        "processing_time_seconds": round(
            end_time - start_time,
            4
        )
    }


# =========================================================
# Unified Prediction Router
# =========================================================

def predict_news(
    text,
    model_type="xgboost"
):

    # Validate input
    if not isinstance(text, str) or not text.strip():

        raise ValueError(
            "Input text must be a non-empty string."
        )

    # XGBoost prediction
    if model_type == "xgboost":

        return predict_xgboost(text)

    # BERT prediction
    elif model_type == "bert":

        return predict_bert(text)

    # Invalid model
    else:

        raise ValueError(
            f"""
            Unsupported model type: {model_type}

            Supported models:
            - xgboost
            - bert
            """
        )


# =========================================================
# Local Testing
# =========================================================

if __name__ == "__main__":

    sample_text = """
    Scientists discover a revolutionary AI system
    for detecting fake news online.
    """

    print("\n==============================")
    print("XGBoost Prediction")
    print("==============================")

    xgb_result = predict_news(
        sample_text,
        model_type="xgboost"
    )

    print(xgb_result)

    print("\n==============================")
    print("BERT Prediction")
    print("==============================")

    bert_result = predict_news(
        sample_text,
        model_type="bert"
    )

    print(bert_result)
