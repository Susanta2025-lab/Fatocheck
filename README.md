## 🚀 Key Project Contributions
- **Built an end-to-end NLP fake news classification pipeline:** 
  Designed and implemented a robust data and modeling workflow, including custom text preprocessing and an optimized TF-IDF feature engineering strategy capturing unigrams and bigrams (up to 10,000 features) to handle sparse text data effectively.

- **Developed, tuned, and optimized tree-based and linear models:** 
  Trained and evaluated multiple estimators, performing hyperparameter optimization using `RandomizedSearchCV`. Successfully deployed Logistic Regression (**96.34% test accuracy**) and Random Forest baselines alongside a high-performance XGBoost model optimized with histogram-based tree methods (`tree_method='hist'`), achieving a peak cross-validation score of **0.9712** and **97.08% test accuracy**.

- **Engineered deployment-ready production pipelines:** 
  Serialized and encapsulated the fitted `TfidfVectorizer` and optimized estimators together using Scikit-Learn `Pipeline` structures. Stored them as standalone `.joblib` objects in a version-controlled modular directory (`models/trained/`) to guarantee strict feature consistency between training and inference phases.

- **Implementing a REST API for real-time inference:** 
  Developing a clean FastAPI service structure within the project architecture (`api/app.py`) to load the serialized pipeline artifacts and serve the optimized XGBoost classifier for real-time text predictions over standard HTTP requests.

- **Containerizing the application for reliable deployment:** 
  Utilizing Docker to containerize the API and utility segments, guaranteeing environment reproducibility, robust portability, and a seamless cloud deployment workflow.
