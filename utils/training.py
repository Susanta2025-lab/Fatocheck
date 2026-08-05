"""
=====================================
FATOCHECK - Model Training Pipeline
utils/training.py
=====================================

This module provides end-to-end training utilities for:
1. Classical ML Pipeline (TF-IDF + XGBoost)
2. BERT Transformer Model Fine-tuning
3. Model serialization and evaluation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd

# Deep Learning
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ML Libraries
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Utils
from utils.preprocessing import TextPreprocessor
from utils.settings import configure_logging, get_settings

# =====================================
# Configuration & Logging
# =====================================
# Module-level path aliases are retained for monkeypatching and callers.
# Directories and logging handlers are NOT created at import time.

_settings = get_settings()
BASE_DIR = _settings.base_dir
MODELS_DIR = _settings.models_dir
DATA_DIR = _settings.data_dir
LOGS_DIR = _settings.logs_dir

logger = logging.getLogger(__name__)


# =====================================
# Classical ML Training
# =====================================


class ClassicalMLTrainer:
    """Train classical ML models (Logistic Regression, Random Forest, XGBoost)"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.metrics = {}

    def load_data(self, filepath: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data"""

        logger.info(f"Loading data from {filepath}")

        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".json"):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")

        # Validate required columns
        if "title" not in df.columns or "text" not in df.columns or "label" not in df.columns:
            raise ValueError("DataFrame must contain 'title', 'text', and 'label' columns")

        logger.info(f"Loaded {len(df)} samples")

        # Preprocess data
        df = self.preprocessor.preprocess_dataframe(df)

        # Split data
        X = df["content"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

        return (X_train, X_test, y_train, y_test)

    def build_xgboost_pipeline(self, **xgb_params) -> Pipeline:
        """Build TF-IDF + XGBoost pipeline"""

        logger.info("Building TF-IDF + XGBoost pipeline")

        # Default XGBoost parameters
        default_params = {
            "n_estimators": 200,
            "max_depth": 7,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }

        # Update with provided params
        default_params.update(xgb_params)

        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8, lowercase=True, stop_words="english"
        )

        # XGBoost classifier
        xgb_clf = xgb.XGBClassifier(**default_params)

        # Create pipeline
        pipeline = Pipeline([("tfidf", vectorizer), ("xgboost", xgb_clf)])

        self.vectorizer = vectorizer
        self.model = pipeline

        return pipeline

    def build_logistic_regression_pipeline(self) -> Pipeline:
        """Build TF-IDF + Logistic Regression pipeline"""

        logger.info("Building TF-IDF + Logistic Regression pipeline")

        vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8, lowercase=True, stop_words="english"
        )

        lr_clf = LogisticRegression(max_iter=1000, random_state=self.random_state, n_jobs=-1)

        pipeline = Pipeline([("tfidf", vectorizer), ("logistic_regression", lr_clf)])

        self.vectorizer = vectorizer
        self.model = pipeline

        return pipeline

    def build_random_forest_pipeline(self) -> Pipeline:
        """Build TF-IDF + Random Forest pipeline"""

        logger.info("Building TF-IDF + Random Forest pipeline")

        vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8, lowercase=True, stop_words="english"
        )

        rf_clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=self.random_state, n_jobs=-1)

        pipeline = Pipeline([("tfidf", vectorizer), ("random_forest", rf_clf)])

        self.vectorizer = vectorizer
        self.model = pipeline

        return pipeline

    def train(self, X_train, y_train):
        """Train the pipeline"""

        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        logger.info("Training completed!")

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate model performance"""

        logger.info("Evaluating model...")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        self.metrics = metrics

        logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

        return metrics

    def save_model(self, filepath: Optional[str] = None):
        """Save the trained model"""

        if filepath is None:
            filepath = MODELS_DIR / "xgboost_pipeline.joblib"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

        # Save metrics
        metrics_file = filepath.parent / f"{filepath.stem}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_file}")

    def load_model(self, filepath: str):
        """Load a trained model"""

        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


# =====================================
# BERT Transformer Training
# =====================================


class BertTrainer:
    """Fine-tune BERT for fake news detection"""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.preprocessor = TextPreprocessor()
        self.metrics = {}

        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Using device: {self.device}")

    def load_data(self, filepath: str, test_size: float = 0.2):
        """Load and preprocess data for BERT"""

        logger.info(f"Loading data from {filepath}")

        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".json"):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format")

        # Preprocess
        df = self.preprocessor.preprocess_dataframe(df)

        X = df["content"].tolist()
        y = df["label"].tolist()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Tokenize
        logger.info("Tokenizing texts...")

        def tokenize_function(texts, labels):
            encodings = self.tokenizer(texts, max_length=self.max_length, truncation=True, padding="max_length")
            encodings["labels"] = labels
            return encodings

        train_encodings = tokenize_function(X_train, y_train)
        test_encodings = tokenize_function(X_test, y_test)

        # Convert to PyTorch datasets
        train_dataset = TensorDataset(
            torch.tensor(train_encodings["input_ids"]),
            torch.tensor(train_encodings["attention_mask"]),
            torch.tensor(train_encodings["labels"]),
        )

        test_dataset = TensorDataset(
            torch.tensor(test_encodings["input_ids"]),
            torch.tensor(test_encodings["attention_mask"]),
            torch.tensor(test_encodings["labels"]),
        )

        return train_dataset, test_dataset

    def train(self, train_dataset, test_dataset):
        """Fine-tune BERT"""

        logger.info("Starting BERT fine-tuning...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=MODELS_DIR / self.model_name,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=LOGS_DIR,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=self.learning_rate,
            seed=self.random_state,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        trainer.train()
        logger.info("Fine-tuning completed!")

        return trainer

    def evaluate(self, trainer, test_dataset):
        """Evaluate BERT model"""

        logger.info("Evaluating BERT model...")

        results = trainer.evaluate(test_dataset)
        self.metrics = results

        logger.info("Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")

        return results

    def save_model(self, filepath: Optional[str] = None):
        """Save fine-tuned BERT model"""

        if filepath is None:
            filepath = MODELS_DIR / self.model_name

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)

        logger.info(f"BERT model saved to {filepath}")

        # Save metrics
        metrics_file = filepath / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)


# =====================================
# Training Pipeline Orchestration
# =====================================


class TrainingPipeline:
    """Orchestrate end-to-end model training"""

    def __init__(self, data_filepath: str):
        self.data_filepath = data_filepath
        self.results = {}

    def train_xgboost(self, **xgb_params) -> Dict[str, Any]:
        """Train XGBoost model"""

        logger.info("=" * 60)
        logger.info("TRAINING: XGBoost Pipeline")
        logger.info("=" * 60)

        trainer = ClassicalMLTrainer()
        X_train, X_test, y_train, y_test = trainer.load_data(self.data_filepath)

        trainer.build_xgboost_pipeline(**xgb_params)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        trainer.save_model()

        self.results["xgboost"] = {"model": trainer.model, "metrics": metrics}

        return metrics

    def train_logistic_regression(self) -> Dict[str, Any]:
        """Train Logistic Regression model"""

        logger.info("=" * 60)
        logger.info("TRAINING: Logistic Regression Pipeline")
        logger.info("=" * 60)

        trainer = ClassicalMLTrainer()
        X_train, X_test, y_train, y_test = trainer.load_data(self.data_filepath)

        trainer.build_logistic_regression_pipeline()
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        trainer.save_model(MODELS_DIR / "logistic_regression_pipeline.joblib")

        self.results["logistic_regression"] = {"model": trainer.model, "metrics": metrics}

        return metrics

    def train_random_forest(self) -> Dict[str, Any]:
        """Train Random Forest model"""

        logger.info("=" * 60)
        logger.info("TRAINING: Random Forest Pipeline")
        logger.info("=" * 60)

        trainer = ClassicalMLTrainer()
        X_train, X_test, y_train, y_test = trainer.load_data(self.data_filepath)

        trainer.build_random_forest_pipeline()
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        trainer.save_model(MODELS_DIR / "random_forest_pipeline.joblib")

        self.results["random_forest"] = {"model": trainer.model, "metrics": metrics}

        return metrics

    def train_bert(
        self, model_name: str = "bert-base-uncased", num_epochs: int = 3, batch_size: int = 8
    ) -> Dict[str, Any]:
        """Train BERT model"""

        logger.info("=" * 60)
        logger.info("TRAINING: BERT Transformer")
        logger.info("=" * 60)

        trainer = BertTrainer(model_name=model_name, num_epochs=num_epochs, batch_size=batch_size)

        train_dataset, test_dataset = trainer.load_data(self.data_filepath)
        bert_trainer = trainer.train(train_dataset, test_dataset)
        metrics = trainer.evaluate(bert_trainer, test_dataset)
        trainer.save_model()

        self.results["bert"] = {"model": trainer.model, "metrics": metrics}

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""

        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)

        summary = {}
        for model_name, result in self.results.items():
            summary[model_name] = result.get("metrics", {})
            logger.info(f"\n{model_name.upper()}:")
            for metric, value in summary[model_name].items():
                logger.info(f"  {metric}: {value:.4f}")

        return summary


# =====================================
# Hyperparameter Tuning
# =====================================
#
# Single source of truth for hyperparameter search. This class was
# previously duplicated with conflicting APIs in both training.py
# (GridSearchCV, XGBoost only) and evaluation.py (RandomizedSearchCV,
# Logistic Regression + Random Forest + XGBoost). The implementation kept
# here is the RandomizedSearchCV version, since it matches the approach
# actually used in notebooks/02_classical_ml.ipynb. utils.evaluation
# re-exports this class so existing imports keep working.


class HyperparameterTuner:
    """Optimize model hyperparameters using RandomizedSearchCV"""

    def __init__(self):
        self.best_params = None
        self.best_score = None
        self.search_results = None
        self.cv_results = None

    def tune_logistic_regression(self, X_train, y_train, n_iter: int = 10, cv: int = 5) -> Dict[str, Any]:
        """
        Tune Logistic Regression with RandomizedSearchCV

        Args:
            X_train: Training features (TF-IDF vectorized)
            y_train: Training labels
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds

        Returns:
            Dictionary with best parameters and score
        """
        logger.info("Starting Logistic Regression hyperparameter tuning...")

        param_distributions = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}

        log_reg = LogisticRegression(solver="liblinear", random_state=42)

        random_search = RandomizedSearchCV(
            estimator=log_reg,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )

        random_search.fit(X_train, y_train)

        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.cv_results = random_search.cv_results_

        logger.info(f"Best Parameters: {self.best_params}")
        logger.info(f"Best Cross-Validation Score: {self.best_score:.4f}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_estimator": random_search.best_estimator_,
        }

    def tune_random_forest(self, X_train, y_train, n_iter: int = 10, cv: int = 5) -> Dict[str, Any]:
        """
        Tune Random Forest with RandomizedSearchCV

        Args:
            X_train: Training features (TF-IDF vectorized)
            y_train: Training labels
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds

        Returns:
            Dictionary with best parameters and score
        """
        logger.info("Starting Random Forest hyperparameter tuning...")

        param_distributions = {
            "n_estimators": [100, 150, 200],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        rf_model = RandomForestClassifier(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )

        random_search.fit(X_train, y_train)

        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.cv_results = random_search.cv_results_

        logger.info(f"Best Parameters: {self.best_params}")
        logger.info(f"Best Cross-Validation Score: {self.best_score:.4f}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_estimator": random_search.best_estimator_,
        }

    def tune_xgboost(self, X_train, y_train, n_iter: int = 10, cv: int = 5) -> Dict[str, Any]:
        """
        Tune XGBoost with RandomizedSearchCV

        Args:
            X_train: Training features (TF-IDF vectorized)
            y_train: Training labels
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds

        Returns:
            Dictionary with best parameters and score
        """
        logger.info("Starting XGBoost hyperparameter tuning...")

        param_distributions = {
            "n_estimators": [100, 150, 200],
            "max_depth": [5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        }

        # NOTE: `use_label_encoder` was dropped here. It was deprecated by
        # xgboost and is unsupported under the pinned xgboost==3.2.0; this
        # tuner path is not invoked by any current model artifact.
        xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )

        random_search.fit(X_train, y_train)

        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.cv_results = random_search.cv_results_

        logger.info(f"Best Parameters: {self.best_params}")
        logger.info(f"Best Cross-Validation Score: {self.best_score:.4f}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_estimator": random_search.best_estimator_,
        }

    def get_cv_results_dataframe(self) -> pd.DataFrame:
        """
        Get cross-validation results as DataFrame

        Returns:
            DataFrame with CV results
        """
        if self.cv_results is None:
            logger.warning("No CV results available")
            return pd.DataFrame()

        df = pd.DataFrame(self.cv_results)
        return df[["param_" + key for key in self.best_params.keys()] + ["mean_test_score", "std_test_score"]]


# =====================================
# Main Training Script
# =====================================

if __name__ == "__main__":

    import argparse

    # Logging is configured only at the CLI entry point, not on import.
    configure_logging(
        log_file=LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    parser = argparse.ArgumentParser(description="Train Fatocheck models")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "logistic_regression", "random_forest", "bert", "all"],
        help="Model to train",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="BERT model variant",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for BERT")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = TrainingPipeline(args.data)

    # Train models
    if args.model in ["xgboost", "all"]:
        pipeline.train_xgboost()

    if args.model in ["logistic_regression", "all"]:
        pipeline.train_logistic_regression()

    if args.model in ["random_forest", "all"]:
        pipeline.train_random_forest()

    if args.model in ["bert", "all"]:
        pipeline.train_bert(model_name=args.bert_model, num_epochs=args.epochs, batch_size=args.batch_size)

    # Get summary
    summary = pipeline.get_summary()

    logger.info("Training pipeline completed successfully!")
