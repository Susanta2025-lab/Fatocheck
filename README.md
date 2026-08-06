# FatoCheck: AI-Powered Fake News Detection

<p align="center">
  An end-to-end NLP and AI engineering project for classifying news articles as fake or real.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--Learn-TF--IDF-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-Production%20Model-337AB7">
  <img src="https://img.shields.io/badge/Transformers-BERT-FFD21E?logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white">
  <a href="https://github.com/Susanta2025-lab/Fatocheck/actions/workflows/ci.yml">
    <img src="https://github.com/Susanta2025-lab/Fatocheck/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/coverage-pytest--cov-lightgrey" alt="Coverage (pytest-cov)">
</p>

<p align="center">
  <a href="https://fatocheck.onrender.com/docs">
    <img src="https://img.shields.io/badge/Live%20API-Render-purple?logo=render&logoColor=white">
  </a>
  <a href="https://www.kaggle.com/code/susantahazra/welfake-news-detection-tuned-xgboost-97">
    <img src="https://img.shields.io/badge/Kaggle-XGBoost%20Notebook-20BEFF?logo=kaggle&logoColor=white">
  </a>
</p>

---

## Overview

FatoCheck is a fake-news detection system that combines:

* NLP data exploration and preprocessing
* TF-IDF feature engineering
* Logistic Regression, Random Forest, and XGBoost
* fine-tuned `bert-base-uncased`
* unified model inference
* FastAPI REST endpoints
* Docker containerization
* Render deployment

The public production API currently uses the tuned **TF-IDF + XGBoost pipeline**.

The fine-tuned BERT classifier is supported and verified for local inference, but its large `model.safetensors` artifact is not included in the GitHub-based Render deployment.

---

## Current System Status

| Component                    | Status                 |
| ---------------------------- | ---------------------- |
| Exploratory data analysis    | Complete               |
| Text preprocessing pipeline  | Complete               |
| Logistic Regression baseline | Complete               |
| Random Forest model          | Complete               |
| Tuned XGBoost model          | Complete               |
| Fine-tuned BERT model        | Working locally        |
| Unified inference layer      | Complete               |
| FastAPI backend              | Complete               |
| Docker container             | Complete               |
| Render deployment            | XGBoost production API |
| BERT cloud deployment        | Planned                |

---

## Model Performance

### Tuned XGBoost

The optimized TF-IDF + XGBoost pipeline achieved approximately:

* **Test accuracy:** 97.08%
* High precision and recall across both classes
* Fast CPU inference suitable for API deployment

The full experiment is available in the Kaggle notebook:

```text
welfake-news-detection-tuned-xgboost-97.ipynb
```

### Fine-Tuned BERT

The transformer experiments use:

```text
bert-base-uncased
```

The repository includes:

* transformer training notebooks
* tokenizer configuration
* fine-tuned model configuration
* local inference support
* confidence scoring
* lazy model loading

The BERT model is loaded only when the first BERT request is made.

Verified local behavior:

```text
Before prediction: bert_loaded = False
After prediction:  bert_loaded = True
```

---

## Production Architecture

```text
Client
  │
  ▼
FastAPI
  │
  ▼
Unified inference router
  │
  ├── XGBoost
  │     └── preprocessing → TF-IDF → classifier
  │
  └── Fine-tuned BERT
        └── preprocessing → tokenizer → transformer
  │
  ▼
Prediction + confidence + processing time
```

The API layer remains thin. Model loading, preprocessing, routing, and prediction logic are implemented in:

```text
utils/inference.py
```

---

## Core Features

* Fake-versus-real news classification
* Combined headline and article input
* Reusable text preprocessing
* TF-IDF unigrams and bigrams
* Multiple classical ML experiments
* Tuned XGBoost production pipeline
* Fine-tuned BERT inference
* Model selection through the API
* Confidence scores
* Processing-time measurement
* Local-only BERT loading
* Lazy transformer initialization
* Health and readiness endpoints
* Interactive Swagger documentation
* Dockerized deployment

---

## Dataset

FatoCheck uses the WELFake dataset.

Main columns:

```text
title
text
label
```

The project uses the following label convention:

```text
0 = Fake
1 = Real
```

Processed data is stored at:

```text
data/processed/cleaned_news.csv
```

---

## NLP Workflow

### Exploratory analysis

The exploration notebook includes:

* missing-value analysis
* duplicate detection and removal
* label verification
* article-length distributions
* word clouds
* frequent-word analysis
* bigram and n-gram analysis
* fake-versus-real text comparisons

### Text preprocessing

The reusable `TextPreprocessor` performs:

* invalid-value handling
* URL removal
* HTML-tag removal
* lowercase conversion
* punctuation and number removal
* whitespace normalization

Example:

```python
from utils.preprocessing import TextPreprocessor

processor = TextPreprocessor()

cleaned_text = processor.clean_text(
    "Breaking NEWS! Read more at https://example.com"
)
```

---

## API Endpoints

| Method | Endpoint   | Purpose                            |
| ------ | ---------- | ---------------------------------- |
| `GET`  | `/`        | Service and model status           |
| `GET`  | `/health`  | Lightweight health check           |
| `GET`  | `/ready`   | Deployment readiness check         |
| `GET`  | `/models`  | Model information and availability |
| `POST` | `/predict` | Run fake-news classification       |

Interactive API documentation:

```text
https://fatocheck.onrender.com/docs
```

---

## Prediction Request

### XGBoost

```json
{
  "title": "Scientists announce a new AI system",
  "text": "Researchers published details of a system for detecting misinformation.",
  "model": "xgboost"
}
```

### BERT — local backend

```json
{
  "title": "Scientists announce a new AI system",
  "text": "Researchers published details of a system for detecting misinformation.",
  "model": "bert"
}
```

## Example Response

```json
{
  "success": true,
  "result": {
    "model": "xgboost",
    "prediction": "Real",
    "confidence": 0.9724,
    "processing_time_seconds": 0.0312
  }
}
```

### Error response

Application errors use a consistent envelope (HTTP status codes are unchanged):

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable message",
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

### Operational headers

Every response includes:

| Header | Purpose |
| ------ | ------- |
| `X-Request-ID` | Correlation ID (echoed from the request when provided, otherwise generated) |
| `X-Process-Time` | Request duration in seconds |
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `no-referrer` |

Pass a client ID when useful:

```bash
curl -H "X-Request-ID: my-trace-1" https://fatocheck.onrender.com/health
```

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Susanta2025-lab/Fatocheck.git
cd Fatocheck
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For an existing pyenv environment:

```bash
pyenv activate Fatocheck
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start the backend

```bash
uvicorn api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Configuration

Runtime settings are centralized in `utils/settings.py` and can be overridden with environment variables or a local `.env` file (see `.env.example`):

| Variable     | Default     | Purpose                                              |
| ------------ | ----------- | ---------------------------------------------------- |
| `MODEL_TYPE` | `xgboost`   | Default inference model when the client omits `model` |
| `LOG_LEVEL`  | `INFO`      | Process log level                                    |
| `API_HOST`   | `0.0.0.0`   | Uvicorn bind host (Docker / compose)                 |
| `API_PORT`   | `8000`      | Uvicorn bind port (Docker / compose)                 |
| `MODELS_DIR` | *(unset)*   | Optional override for `models/trained`               |

Per-request model selection via the API `model` field still overrides `MODEL_TYPE`.


### Streamlit UI (local demo)

The Streamlit app is a presentation layer only. It never loads models; it calls the FastAPI backend over HTTP.

```text
Browser
   │
   ▼
Streamlit UI
   │  HTTP
   ▼
FastAPI
   │
Inference Layer
   ├── XGBoost (production)
   └── BERT (optional / local)
```

1. Start the API:

```bash
uvicorn api.app:app --reload --port 8000
```

2. In a second terminal, start Streamlit:

```bash
streamlit run streamlit_app.py
```

Optional environment variable:

```text
FATOCHECK_API_URL=http://127.0.0.1:8000
```

Point `FATOCHECK_API_URL` at any reachable FatoCheck API (local or remote). Do not hard-code deployment URLs in the UI.


### Continuous Integration

GitHub Actions runs on pushes and pull requests to `master` (workflow: `.github/workflows/ci.yml`):

1. lint, format check, unit tests, and coverage
2. Docker image build (`fatocheck-api:ci`) after quality checks pass

Run the same quality checks locally:

```bash
python -m pip install -r requirements-dev.txt
make lint
make format-check
pytest tests/ -v --cov=api --cov=utils --cov-report=term-missing --cov-report=xml:coverage.xml
```

Coverage is produced with `pytest-cov` in CI (`coverage.xml`). There is no external coverage hosting service configured yet; the README coverage badge is a placeholder.

---

## Test the Inference Layer

### Health status

```bash
python -c "
from utils.inference import health_check
print(health_check())
"
```

### XGBoost prediction

```bash
python -c "
from utils.inference import predict_news

print(
    predict_news(
        'Scientists developed a new AI model for detecting misinformation.',
        model_type='xgboost'
    )
)
"
```

### BERT prediction

BERT requires the local fine-tuned model weights.

```bash
python -c "
from utils.inference import predict_news

print(
    predict_news(
        'Scientists developed a new AI model for detecting misinformation.',
        model_type='bert'
    )
)
"
```

---

## Run with Docker

### Build the image

```bash
docker build -t fatocheck-api .
```

### Run the container

```bash
docker run --rm -p 8000:8000 fatocheck-api
```

Open:

```text
http://localhost:8000/docs
```

---

## Project Structure

```text
.
├── api
│   └── app.py
├── data
│   ├── processed
│   │   └── cleaned_news.csv
│   └── raw
│       ├── archive.zip
│       └── WELFake_Dataset.csv
├── Dockerfile
├── .env.example
├── main.py
├── Makefile
├── models
│   └── trained
│       ├── bert-base-uncased
│       ├── logistic_regression_pipeline.joblib
│       ├── random_forest_pipeline.joblib
│       └── xgboost_pipeline.joblib
├── notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_transformer.ipynb
│   └── archive/
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── results
└── utils
    ├── evaluation.py
    ├── inference.py
    ├── __init__.py
    ├── preprocessing.py
    ├── settings.py
    └── training.py
```

---

## Engineering Decisions

### Serialized classical pipelines

The TF-IDF vectorizer and estimator are stored together in Scikit-Learn pipelines. This prevents differences between training-time and inference-time feature processing.

### Unified inference router

The same inference module can route requests to:

```text
xgboost
bert
```

### Lazy BERT loading

BERT is not loaded during application startup. The model is initialized during the first BERT request and remains in memory for the lifetime of that backend process.

### Local-only transformer loading

Production inference uses:

```python
local_files_only=True
```

This prevents the backend from silently downloading the untouched public BERT checkpoint instead of loading the fine-tuned FatoCheck model.

### Thin FastAPI layer

`api/app.py` handles:

* request validation
* endpoint routing
* HTTP responses
* service health information

Model logic remains in `utils/inference.py`.

---

## BERT Deployment Limitation

The fine-tuned BERT weight file is too large for normal GitHub storage.

Therefore:

* BERT inference works locally when the model artifact is present.
* The GitHub-based Render deployment currently serves XGBoost.
* BERT model hosting will later use Git LFS, Hugging Face Hub, or cloud object storage.

This keeps the public API lightweight and reliable while preserving the transformer implementation as part of the project.

---

## Limitations

* FatoCheck predicts learned linguistic and source patterns; it does not independently verify factual claims.
* High confidence does not guarantee factual correctness.
* Performance may decrease for recent news, unfamiliar domains, satire, non-English text, or adversarial content.
* Dataset-specific patterns may affect model generalization.
* BERT predictions may be overconfident.
* The system should not be used as the only basis for journalistic, political, legal, or safety-critical decisions.

---

## Roadmap

* Host the fine-tuned BERT artifact externally
* Deploy transformer inference separately
* Add automated API and inference tests
* Add GitHub Actions CI
* Add confidence calibration
* Add batch CSV prediction
* Add a Streamlit frontend
* Add model monitoring and drift detection
* Add evidence retrieval and source-backed fact checking

---

## Author

**Susanta Hazra**

AI Engineer with a PhD and a research background in chemical science, building production-oriented machine learning, NLP, RAG, and AI systems.

* GitHub: `Susanta2025-lab`
* Kaggle: `susantahazra`

---

## Disclaimer

FatoCheck is an educational and portfolio project. Its output is a statistical model prediction and should be combined with evidence review, source verification, and human judgment.
