"""Tests for utils/training.py.

These tests exercise configuration and orchestration logic without running
real model training or real hyperparameter search workloads.
"""

import importlib

import pytest


class _DummyFittedModel:
    """Minimal stand-in that joblib can pickle in place of a real model."""

    def predict(self, X):
        return [0 for _ in X]


class TestHyperparameterTuner:
    def test_exposes_expected_tuning_methods(self):
        from utils.training import HyperparameterTuner

        tuner = HyperparameterTuner()

        assert hasattr(tuner, "tune_logistic_regression")
        assert hasattr(tuner, "tune_random_forest")
        assert hasattr(tuner, "tune_xgboost")

    def test_evaluation_reexports_same_class(self):
        import utils.evaluation as evaluation
        import utils.training as training

        assert evaluation.HyperparameterTuner is training.HyperparameterTuner

    def test_tune_logistic_regression_uses_randomized_search(self, monkeypatch):
        import utils.training as training

        class FakeRandomizedSearchCV:
            def __init__(self, estimator, param_distributions, **kwargs):
                self.estimator = estimator
                self.best_params_ = {"C": 1, "penalty": "l2"}
                self.best_score_ = 0.95
                self.best_estimator_ = estimator
                self.cv_results_ = {"mean_test_score": [0.95]}

            def fit(self, X, y):
                return self

        monkeypatch.setattr(training, "RandomizedSearchCV", FakeRandomizedSearchCV)

        tuner = training.HyperparameterTuner()
        result = tuner.tune_logistic_regression(
            X_train=[[1], [2], [3]], y_train=[0, 1, 0], n_iter=2, cv=2
        )

        assert result["best_params"] == {"C": 1, "penalty": "l2"}
        assert result["best_score"] == 0.95

    def test_tune_random_forest_uses_randomized_search(self, monkeypatch):
        import utils.training as training

        class FakeRandomizedSearchCV:
            def __init__(self, estimator, param_distributions, **kwargs):
                self.best_params_ = {"n_estimators": 100}
                self.best_score_ = 0.9
                self.best_estimator_ = estimator
                self.cv_results_ = {"mean_test_score": [0.9]}

            def fit(self, X, y):
                return self

        monkeypatch.setattr(training, "RandomizedSearchCV", FakeRandomizedSearchCV)

        tuner = training.HyperparameterTuner()
        result = tuner.tune_random_forest(
            X_train=[[1], [2], [3]], y_train=[0, 1, 0], n_iter=2, cv=2
        )

        assert result["best_params"] == {"n_estimators": 100}

    def test_tune_xgboost_uses_randomized_search(self, monkeypatch):
        import utils.training as training

        class FakeRandomizedSearchCV:
            def __init__(self, estimator, param_distributions, **kwargs):
                self.best_params_ = {"max_depth": 5}
                self.best_score_ = 0.93
                self.best_estimator_ = estimator
                self.cv_results_ = {"mean_test_score": [0.93]}

            def fit(self, X, y):
                return self

        monkeypatch.setattr(training, "RandomizedSearchCV", FakeRandomizedSearchCV)

        tuner = training.HyperparameterTuner()
        result = tuner.tune_xgboost(
            X_train=[[1], [2], [3]], y_train=[0, 1, 0], n_iter=2, cv=2
        )

        assert result["best_params"] == {"max_depth": 5}


class TestClassicalMLTrainerConfiguration:
    @pytest.mark.parametrize(
        "builder_name",
        [
            "build_xgboost_pipeline",
            "build_logistic_regression_pipeline",
            "build_random_forest_pipeline",
        ],
    )
    def test_pipelines_preserve_tfidf_settings(self, builder_name):
        from utils.training import ClassicalMLTrainer

        trainer = ClassicalMLTrainer()
        pipeline = getattr(trainer, builder_name)()
        vectorizer = pipeline.named_steps["tfidf"]

        assert vectorizer.max_features == 10000
        assert vectorizer.ngram_range == (1, 2)
        assert vectorizer.min_df == 5
        assert vectorizer.max_df == 0.8
        assert vectorizer.lowercase is True
        assert vectorizer.stop_words == "english"

    def test_xgboost_pipeline_default_hyperparameters(self):
        from utils.training import ClassicalMLTrainer

        trainer = ClassicalMLTrainer()
        pipeline = trainer.build_xgboost_pipeline()
        xgb_params = pipeline.named_steps["xgboost"].get_params()

        assert xgb_params["n_estimators"] == 200
        assert xgb_params["max_depth"] == 7
        assert xgb_params["learning_rate"] == 0.1


class TestModelArtifactFilenames:
    def test_default_xgboost_artifact_filenames(self, tmp_path, monkeypatch):
        import utils.training as training
        from utils.training import ClassicalMLTrainer

        monkeypatch.setattr(training, "MODELS_DIR", tmp_path)

        trainer = ClassicalMLTrainer()
        trainer.model = _DummyFittedModel()
        trainer.metrics = {"accuracy": 1.0}

        trainer.save_model()

        assert (tmp_path / "xgboost_pipeline.joblib").exists()
        assert (tmp_path / "xgboost_pipeline_metrics.json").exists()

    def test_save_model_creates_parent_dir_only_when_invoked(self, tmp_path):
        from utils.training import ClassicalMLTrainer

        trainer = ClassicalMLTrainer()
        trainer.model = _DummyFittedModel()
        trainer.metrics = {}

        target_dir = tmp_path / "nested" / "dir"
        assert not target_dir.exists()

        trainer.save_model(target_dir / "model.joblib")

        assert target_dir.exists()
        assert (target_dir / "model.joblib").exists()

    def test_training_pipeline_uses_expected_artifact_filenames(self, monkeypatch):
        import utils.training as training

        captured = {}

        class DummyTrainer:
            def __init__(self, *args, **kwargs):
                self.model = _DummyFittedModel()

            def load_data(self, filepath):
                return ([], [], [], [])

            def build_logistic_regression_pipeline(self):
                return None

            def build_random_forest_pipeline(self):
                return None

            def build_xgboost_pipeline(self, **kwargs):
                return None

            def train(self, X_train, y_train):
                pass

            def evaluate(self, X_test, y_test):
                return {}

            def save_model(self, filepath=None):
                captured["filepath"] = filepath

        monkeypatch.setattr(training, "ClassicalMLTrainer", DummyTrainer)

        pipeline = training.TrainingPipeline("dummy.csv")

        pipeline.train_logistic_regression()
        assert str(captured["filepath"]).endswith("logistic_regression_pipeline.joblib")

        pipeline.train_random_forest()
        assert str(captured["filepath"]).endswith("random_forest_pipeline.joblib")


class TestNoImportTimeSideEffects:
    def test_importing_training_creates_no_directories_or_logs(self):
        import utils.training as training

        logs_existed_before = training.LOGS_DIR.exists()
        models_existed_before = training.MODELS_DIR.exists()

        importlib.reload(training)

        if not logs_existed_before:
            assert not training.LOGS_DIR.exists()

        if not models_existed_before:
            assert not training.MODELS_DIR.exists()
