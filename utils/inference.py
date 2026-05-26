import joblib
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocessing import TextPreprocessor


# =========================================================
# Initialize Preprocessor
# =========================================================

processor = TextPreprocessor()


# =========================================================
# Base Directory
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]


# =========================================================
# Model Paths
# =========================================================

MODEL_PATHS = {

    "logistic_regression":
        BASE_DIR / "models" / "trained" / "logistic_regression_pipeline.joblib",

    "random_forest":
        BASE_DIR / "models" / "trained" / "random_forest_pipeline.joblib",

    "xgboost":
        BASE_DIR / "models" / "trained" / "xgboost_pipeline.joblib"
}


TRANSFORMER_PATH = (
    BASE_DIR / "models" / "trained" / "bert-base-uncased"
)

# =========================================================
# Load Classical ML Models
# =========================================================

loaded_models = {}

for model_name, model_path in MODEL_PATHS.items():
    loaded_models[model_name] = joblib.load(model_path)


# =========================================================
# Load Transformer Model + Tokenizer
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_PATH)

transformer_model = AutoModelForSequenceClassification.from_pretrained(
    TRANSFORMER_PATH
)

transformer_model.eval()


# =========================================================
# Label Mapping
# =========================================================

LABEL_MAP = {
    0: "Fake",
    1: "Real"
}


# =========================================================
# Classical ML Prediction Function
# =========================================================

def predict_classical(text, model_name="xgboost"):

    start_time = time.time()

    # Clean text
    cleaned_text = processor.clean_text(text)

    # Get model
    model = loaded_models[model_name]

    # Prediction
    prediction = model.predict([cleaned_text])[0]

    # Confidence score
    probabilities = model.predict_proba([cleaned_text])[0]

    confidence = round(float(max(probabilities)), 4)

    end_time = time.time()

    return {
        "model": model_name,
        "prediction": LABEL_MAP[prediction],
        "confidence": confidence,
        "processing_time_seconds": round(end_time - start_time, 4)
    }


# =========================================================
# Transformer Prediction Function
# =========================================================

def predict_transformer(text):

    start_time = time.time()

    # Clean text
    cleaned_text = processor.clean_text(text)

    # Tokenize
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Disable gradient calculation
    with torch.no_grad():

        outputs = transformer_model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    end_time = time.time()

    return {
        "model": "bert-base-uncased",
        "prediction": LABEL_MAP[prediction.item()],
        "confidence": round(confidence.item(), 4),
        "processing_time_seconds": round(end_time - start_time, 4)
    }


# =========================================================
# Unified Prediction Router
# =========================================================

def predict_news(text, model_type="xgboost"):

    # Validate input
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")

    # Classical ML models
    if model_type in MODEL_PATHS:
        return predict_classical(text, model_type)

    # Transformer model
    elif model_type == "bert":
        return predict_transformer(text)

    # Invalid model
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )


# =========================================================
# Example Local Testing
# =========================================================

if __name__ == "__main__":

    sample_text = """
    Scientists discover a new method to detect fake news using AI.
    """

    # XGBoost prediction
    result_xgb = predict_news(
        sample_text,
        model_type="xgboost"
    )

    print("\nXGBoost Prediction:")
    print(result_xgb)

    # BERT prediction
    result_bert = predict_news(
        sample_text,
        model_type="bert"
    )

    print("\nBERT Prediction:")
    print(result_bert)
