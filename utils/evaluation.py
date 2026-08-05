"""
=====================================
FATOCHECK - Model Evaluation Framework
utils/evaluation.py
=====================================

Comprehensive evaluation utilities for:
1. Binary classification metrics (Accuracy, Precision, Recall, F1)
2. Confusion Matrix & Classification Reports
3. Model comparison & ranking
4. Comprehensive visualizations

Hyperparameter tuning (RandomizedSearchCV) lives in utils.training.
HyperparameterTuner is re-exported from here for backward compatibility.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ML & Statistics Libraries
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Project utilities
from utils.preprocessing import TextPreprocessor
from utils.settings import configure_logging, get_settings

# HyperparameterTuner now lives in utils.training (single source of truth
# for training/tuning logic); re-exported here for backward compatibility
# with any existing `from utils.evaluation import HyperparameterTuner`.
from utils.training import HyperparameterTuner  # noqa: F401

# =====================================
# Configuration & Logging
# =====================================
# Module-level path aliases are retained for monkeypatching and callers.
# Directories and logging handlers are NOT created at import time.

_settings = get_settings()
BASE_DIR = _settings.base_dir
MODELS_DIR = _settings.models_dir
RESULTS_DIR = _settings.results_dir
PLOTS_DIR = _settings.plots_dir
LOGS_DIR = _settings.logs_dir

logger = logging.getLogger(__name__)


# =====================================
# Data Classes for Results
# =====================================


@dataclass
class ClassificationMetrics:
    """Container for classification metrics"""

    accuracy: float
    precision_fake: float
    recall_fake: float
    f1_fake: float
    precision_real: float
    recall_real: float
    f1_real: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "accuracy": self.accuracy,
            "precision_fake": self.precision_fake,
            "recall_fake": self.recall_fake,
            "f1_fake": self.f1_fake,
            "precision_real": self.precision_real,
            "recall_real": self.recall_real,
            "f1_real": self.f1_real,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
        }

    def __str__(self) -> str:
        """Pretty print metrics"""
        return f"""
Classification Metrics:
  Accuracy:          {self.accuracy:.4f}

  FAKE NEWS (Class 0):
    Precision:       {self.precision_fake:.4f}
    Recall:          {self.recall_fake:.4f}
    F1-Score:        {self.f1_fake:.4f}

  REAL NEWS (Class 1):
    Precision:       {self.precision_real:.4f}
    Recall:          {self.recall_real:.4f}
    F1-Score:        {self.f1_real:.4f}

  WEIGHTED AVERAGE:
    Precision:       {self.precision_weighted:.4f}
    Recall:          {self.recall_weighted:.4f}
    F1-Score:        {self.f1_weighted:.4f}
        """


@dataclass
class ConfusionMatrixResult:
    """Container for confusion matrix data"""

    tn: int  # True Negatives (Fake correctly classified)
    fp: int  # False Positives (Real classified as Fake)
    fn: int  # False Negatives (Fake classified as Real)
    tp: int  # True Positives (Real correctly classified)
    matrix: np.ndarray

    def __str__(self) -> str:
        """Pretty print confusion matrix"""
        return f"""
Confusion Matrix:
              Predicted Fake  Predicted Real
Actual Fake   {self.tn:<15} {self.fp:<15}
Actual Real   {self.fn:<15} {self.tp:<15}
        """


# =====================================
# Core Evaluation Class
# =====================================


class ModelEvaluator:
    """Comprehensive model evaluation framework"""

    def __init__(self, model=None, model_name: str = "model"):
        """
        Initialize evaluator

        Args:
            model: Trained model instance
            model_name: Name of the model for reporting
        """
        self.model = model
        self.model_name = model_name
        self.preprocessor = TextPreprocessor()
        self.metrics = None
        self.confusion_matrix_result = None

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
        """
        Compute classification metrics with per-class breakdown

        Args:
            y_true: True labels
            y_pred: Predicted labels (0=Fake, 1=Real)

        Returns:
            ClassificationMetrics object
        """
        logger.info(f"Computing metrics for {self.model_name}")

        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()

        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class breakdown (0=Fake, 1=Real)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Weighted metrics
        precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        metrics = ClassificationMetrics(
            accuracy=accuracy,
            precision_fake=precision_per_class[0],
            recall_fake=recall_per_class[0],
            f1_fake=f1_per_class[0],
            precision_real=precision_per_class[1],
            recall_real=recall_per_class[1],
            f1_real=f1_per_class[1],
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            f1_weighted=f1_weighted,
        )

        self.metrics = metrics
        logger.info(f"Metrics computed:\n{metrics}")

        return metrics

    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrixResult:
        """
        Compute confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            ConfusionMatrixResult object
        """
        logger.info(f"Computing confusion matrix for {self.model_name}")

        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()

        cm = confusion_matrix(y_true, y_pred)

        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        result = ConfusionMatrixResult(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), matrix=cm)

        self.confusion_matrix_result = result
        logger.info(f"Confusion matrix:\n{result}")

        return result

    def generate_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str] = None
    ) -> str:
        """
        Generate detailed classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Class names ['Fake', 'Real']

        Returns:
            Classification report string
        """
        if target_names is None:
            target_names = ["Fake", "Real"]

        report = classification_report(y_true, y_pred, target_names=target_names)
        logger.info(f"Classification Report:\n{report}")

        return report

    def save_results(self, filepath: Optional[str] = None) -> str:
        """
        Save evaluation results to JSON

        Args:
            filepath: Optional filepath for results

        Returns:
            Path to saved results file
        """
        if filepath is None:
            filepath = RESULTS_DIR / f"{self.model_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "confusion_matrix": (
                {
                    "tn": self.confusion_matrix_result.tn,
                    "fp": self.confusion_matrix_result.fp,
                    "fn": self.confusion_matrix_result.fn,
                    "tp": self.confusion_matrix_result.tp,
                }
                if self.confusion_matrix_result
                else {}
            ),
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {filepath}")

        return str(filepath)


# =====================================
# Visualization Functions
# =====================================


class EvaluationVisualizer:
    """Create publication-quality evaluation plots"""

    def __init__(self, output_dir: Path = PLOTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

    def plot_confusion_matrix(
        self,
        confusion_matrix_result: ConfusionMatrixResult,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (8, 6),
    ) -> str:
        """
        Plot confusion matrix heatmap

        Args:
            confusion_matrix_result: ConfusionMatrixResult object
            model_name: Model name for title
            figsize: Figure size

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=figsize)

        cm = confusion_matrix_result.matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            ax=ax,
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
        )

        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)

        filepath = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix plot saved to {filepath}")

        return str(filepath)

    def plot_metrics_comparison(
        self, metrics_dict: Dict[str, ClassificationMetrics], figsize: Tuple[int, int] = (12, 6)
    ) -> str:
        """
        Plot metrics comparison across multiple models

        Args:
            metrics_dict: Dictionary mapping model names to ClassificationMetrics
            figsize: Figure size

        Returns:
            Path to saved plot
        """
        # Convert to DataFrame
        data = []
        for model_name, metrics in metrics_dict.items():
            row = metrics.to_dict()
            row["Model"] = model_name
            data.append(row)

        df = pd.DataFrame(data)

        # Select key metrics
        metrics_to_plot = ["accuracy", "f1_fake", "f1_real", "f1_weighted"]
        plot_data = df[["Model"] + metrics_to_plot].set_index("Model")

        fig, ax = plt.subplots(figsize=figsize)
        plot_data.plot(kind="bar", ax=ax, width=0.8)

        ax.set_title("Model Metrics Comparison", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylim([0.9, 1.0])
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right")

        filepath = self.output_dir / "metrics_comparison.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Metrics comparison plot saved to {filepath}")

        return str(filepath)


# =====================================
# Model Comparison Class
# =====================================


class ModelComparator:
    """Compare performance of multiple models"""

    def __init__(self):
        self.results = {}
        self.visualizer = EvaluationVisualizer()

    def evaluate_model(self, model, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> ClassificationMetrics:
        """
        Evaluate a single model

        Args:
            model: Trained model
            model_name: Model name for reporting
            X_test: Test features
            y_test: Test labels

        Returns:
            ClassificationMetrics object
        """
        logger.info(f"Evaluating {model_name}")

        evaluator = ModelEvaluator(model, model_name)

        # Get predictions
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = evaluator.compute_metrics(y_test, y_pred)
        evaluator.compute_confusion_matrix(y_test, y_pred)

        # Save results
        self.results[model_name] = {"evaluator": evaluator, "metrics": metrics, "y_pred": y_pred}

        return metrics

    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models

        Returns:
            DataFrame with comparison results
        """
        data = []
        for model_name, result in self.results.items():
            metrics_dict = result["metrics"].to_dict()
            metrics_dict["Model"] = model_name
            data.append(metrics_dict)

        df = pd.DataFrame(data)
        df = df.set_index("Model")

        logger.info(f"Model Comparison:\n{df}")

        return df

    def rank_models(self, metric: str = "f1_weighted") -> pd.DataFrame:
        """
        Rank models by a specific metric

        Args:
            metric: Metric to rank by

        Returns:
            Ranked DataFrame
        """
        data = []
        for model_name, result in self.results.items():
            metrics_dict = result["metrics"].to_dict()
            metrics_dict["Model"] = model_name
            data.append(metrics_dict)

        df = pd.DataFrame(data)
        df_ranked = df.sort_values(metric, ascending=False).reset_index(drop=True)
        df_ranked["Rank"] = df_ranked.index + 1

        logger.info(f"Models ranked by {metric}:\n{df_ranked}")

        return df_ranked

    def save_comparison(self, filepath: Optional[str] = None) -> str:
        """
        Save comparison results

        Args:
            filepath: Optional filepath

        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = RESULTS_DIR / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df = self.compare_models()
        df.to_csv(filepath)

        logger.info(f"Comparison saved to {filepath}")

        return str(filepath)


# =====================================
# Comprehensive Evaluation Pipeline
# =====================================


class EvaluationPipeline:
    """Orchestrate end-to-end evaluation"""

    def __init__(self):
        self.comparator = ModelComparator()
        self.visualizer = EvaluationVisualizer()

    def evaluate_all(
        self, models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, ClassificationMetrics]:
        """
        Evaluate multiple models

        Args:
            models_dict: Dictionary mapping model names to model instances
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)

        results = {}

        for model_name, model in models_dict.items():
            logger.info(f"\nEvaluating {model_name}")

            evaluator = ModelEvaluator(model, model_name)

            # Get predictions
            y_pred = model.predict(X_test)

            # Compute metrics
            metrics = evaluator.compute_metrics(y_test, y_pred)
            cm = evaluator.compute_confusion_matrix(y_test, y_pred)

            # Generate report
            evaluator.generate_classification_report(y_test, y_pred)

            # Create visualizations
            self.visualizer.plot_confusion_matrix(cm, model_name)

            # Save results
            evaluator.save_results()

            results[model_name] = metrics

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)

        return results

    def compare_and_save(self) -> str:
        """
        Compare all models and save results

        Returns:
            Path to comparison results
        """
        self.comparator.compare_models()
        ranked_df = self.comparator.rank_models("f1_weighted")

        # Save to CSV
        comparison_file = self.comparator.save_comparison()

        logger.info(f"\nModel Ranking (by F1-Score):\n{ranked_df}")

        return comparison_file


# =====================================
# Main Script
# =====================================

if __name__ == "__main__":

    import argparse

    # Logging is configured only at the CLI entry point, not on import.
    configure_logging(
        log_file=LOGS_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    parser = argparse.ArgumentParser(description="Evaluate Fatocheck models")
    parser.add_argument("--data", type=str, help="Path to test data")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["xgboost", "logistic_regression"],
        help="Models to evaluate",
    )
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument(
        "--tuner",
        type=str,
        choices=["lr", "rf", "xgb"],
        help="Model to tune using RandomizedSearchCV",
    )
    parser.add_argument("--n_iter", type=int, default=10, help="Iterations for RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")

    args = parser.parse_args()

    logger.info("Evaluation pipeline initialized")
    logger.info(f"Arguments: {args}")

    # Example usage with tuning
    if args.tune and args.tuner:
        tuner = HyperparameterTuner()

        if args.tuner == "lr":
            logger.info("Tuning Logistic Regression...")
            # tuner.tune_logistic_regression(X_train_tfidf, y_train, n_iter=args.n_iter, cv=args.cv)
        elif args.tuner == "rf":
            logger.info("Tuning Random Forest...")
            # tuner.tune_random_forest(X_train_tfidf, y_train, n_iter=args.n_iter, cv=args.cv)
        elif args.tuner == "xgb":
            logger.info("Tuning XGBoost...")
            # tuner.tune_xgboost(X_train_tfidf, y_train, n_iter=args.n_iter, cv=args.cv)
