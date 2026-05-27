# FatoCheck 🔍: Real-Time Fake News Detection Pipeline
An end-to-end NLP project designed to detect misinformation by leveraging a multi-tiered approach—from classical machine learning pipelines to state-of-the-art Transformer architectures.


## 🚀 Key Project Contributions

* **Built an end-to-end NLP fake news classification pipeline:**
Designed and implemented a modular data and modeling workflow, including custom text preprocessing and an optimized TF-IDF feature engineering strategy with unigrams and bigrams (up to 10,000 features) to effectively handle high-dimensional sparse text data.
* **Developed, tuned, and optimized multi-tiered hybrid model architectures:**
Engineered a two-version modeling strategy: Version 1 delivers strong classical ML performance using Logistic Regression, Random Forest, and XGBoost (**97.08% test accuracy**); Version 2 integrates Transformer-based deep learning with `bert-base-uncased`, achieving superior results with **99.17% accuracy** and **99.24% F1-score**.
* **Engineered versioned, deployment-ready production pipelines:**
Serialized and encapsulated fitted vectorizers and estimators into robust Scikit-Learn `Pipeline` structures. Managed separate artifact formats (`.joblib` for classical models and `safetensors` for Transformers) within a version-controlled modular directory (`models/trained/`), ensuring strict consistency between training and inference.
* **Implemented scalable REST API services for real-time inference:**
Developed dual FastAPI service architectures (`api/app-v1.py` and `api/app-v2.py`) to load version-specific model artifacts and serve low-latency classification predictions over standard HTTP requests.
* **Containerized the multi-version application for reliable production deployment:**
Created dedicated Docker configurations (`Dockerfile.v1`, `Dockerfile.v2`) to containerize each inference stack, ensuring environment reproducibility, platform portability, and a streamlined MLOps workflow from local development to cloud-ready deployment.
