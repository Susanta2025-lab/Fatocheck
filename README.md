## 🚀 Key Project Contributions

* **Built an end-to-end NLP fake news classification pipeline:**
Designed and implemented a robust data and modeling workflow, including custom text preprocessing and an optimized TF-IDF feature engineering strategy capturing unigrams and bigrams (up to 10,000 features) to handle sparse text data effectively.
* **Developed, tuned, and optimized tree-based and linear models:**
Trained and evaluated multiple estimators, performing hyperparameter optimization using `RandomizedSearchCV`. Successfully deployed Logistic Regression (**96.34% test accuracy**) and Random Forest baselines alongside a high-performance XGBoost model optimized with histogram-based tree methods (`tree_method='hist'`), achieving a peak cross-validation score of **0.9712** and **97.08% test accuracy**.
* **Engineered deployment-ready production pipelines:**
Serialized and encapsulated the fitted `TfidfVectorizer` and optimized estimators together using Scikit-Learn `Pipeline` structures. Stored them as standalone `.joblib` objects in a version-controlled modular directory (`models/trained/`) to guarantee strict feature consistency between training and inference phases.
* **Implemented a REST API for real-time inference:**
Developed clean, scalable FastAPI service structures within the project architecture (`api/app.py`) to load serialized pipeline artifacts and serve the optimized XGBoost classifier for low-latency text predictions over standard HTTP requests.
* **Containerized the application for reliable production deployment:**
Utilized Docker to containerize the entire API inference stack, ensuring environment reproducibility, robust portability across platforms, and a streamlined MLOps workflow from development to cloud-native production environments.
