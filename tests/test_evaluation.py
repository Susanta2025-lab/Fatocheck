"""Tests for utils/evaluation.py.

Uses small known arrays and temporary directories only; no real plots of
meaningful size are generated and no files are written outside tmp_path.
"""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from utils.evaluation import (
    ClassificationMetrics,
    ConfusionMatrixResult,
    EvaluationVisualizer,
    ModelComparator,
    ModelEvaluator,
)


@pytest.fixture
def known_labels():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    return y_true, y_pred


class _DummyModel:
    def __init__(self, predictions):
        self._predictions = np.array(predictions)

    def predict(self, X):
        return self._predictions


class TestModelEvaluatorMetrics:
    def test_compute_metrics_with_known_arrays(self, known_labels):
        y_true, y_pred = known_labels
        evaluator = ModelEvaluator(model_name="unit-test-model")

        metrics = evaluator.compute_metrics(y_true, y_pred)

        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy == pytest.approx(4 / 6)
        assert 0.0 <= metrics.f1_weighted <= 1.0

    def test_compute_confusion_matrix_shape(self, known_labels):
        y_true, y_pred = known_labels
        evaluator = ModelEvaluator(model_name="unit-test-model")

        result = evaluator.compute_confusion_matrix(y_true, y_pred)

        assert isinstance(result, ConfusionMatrixResult)
        assert result.matrix.shape == (2, 2)
        assert result.tn + result.fp + result.fn + result.tp == len(y_true)

    def test_generate_classification_report_structure(self, known_labels):
        y_true, y_pred = known_labels
        evaluator = ModelEvaluator(model_name="unit-test-model")

        report = evaluator.generate_classification_report(y_true, y_pred)

        assert "Fake" in report
        assert "Real" in report
        assert "precision" in report

    def test_save_results_serialization(self, tmp_path, known_labels):
        y_true, y_pred = known_labels
        evaluator = ModelEvaluator(model_name="unit-test-model")
        evaluator.compute_metrics(y_true, y_pred)
        evaluator.compute_confusion_matrix(y_true, y_pred)

        target = tmp_path / "results.json"
        saved_path = evaluator.save_results(target)

        assert saved_path == str(target)
        assert target.exists()

        with open(target) as f:
            payload = json.load(f)

        assert payload["model_name"] == "unit-test-model"
        assert "accuracy" in payload["metrics"]
        assert "tn" in payload["confusion_matrix"]


class TestModelComparator:
    """Bypasses ModelComparator.__init__ on purpose.

    The real __init__ eagerly constructs an EvaluationVisualizer(), which
    creates the project's results/evaluation/plots directory as a side
    effect. compare_models()/rank_models() never touch that visualizer, so
    building the object with __new__ and setting `results` directly keeps
    these tests from writing outside tmp_path without needing any
    production code changes.
    """

    def test_rank_models_orders_by_metric(self):
        comparator = ModelComparator.__new__(ModelComparator)
        comparator.results = {}

        y_test = np.array([0, 0, 1, 1])
        good_model = _DummyModel([0, 0, 1, 1])
        bad_model = _DummyModel([1, 1, 0, 0])

        for name, model in [("good_model", good_model), ("bad_model", bad_model)]:
            evaluator = ModelEvaluator(model, name)
            y_pred = model.predict(None)
            metrics = evaluator.compute_metrics(y_test, y_pred)
            comparator.results[name] = {
                "evaluator": evaluator,
                "metrics": metrics,
                "y_pred": y_pred,
            }

        ranked = comparator.rank_models("f1_weighted")

        assert ranked.iloc[0]["Model"] == "good_model"
        assert ranked.iloc[0]["Rank"] == 1
        assert ranked.iloc[-1]["Model"] == "bad_model"

    def test_compare_models_returns_all_models(self):
        comparator = ModelComparator.__new__(ModelComparator)
        comparator.results = {}

        y_test = np.array([0, 1, 0, 1])
        model = _DummyModel([0, 1, 0, 1])
        evaluator = ModelEvaluator(model, "only_model")
        y_pred = model.predict(None)
        metrics = evaluator.compute_metrics(y_test, y_pred)
        comparator.results["only_model"] = {
            "evaluator": evaluator,
            "metrics": metrics,
            "y_pred": y_pred,
        }

        df = comparator.compare_models()

        assert list(df.index) == ["only_model"]
        assert df.loc["only_model", "accuracy"] == pytest.approx(1.0)


class TestEvaluationVisualizer:
    def test_plot_confusion_matrix_writes_only_when_called(self, tmp_path):
        import matplotlib.pyplot as plt

        output_dir = tmp_path / "plots"
        visualizer = EvaluationVisualizer(output_dir=output_dir)

        # The directory is created on construction (existing behavior); no
        # plot file should exist until a plot method is actually invoked.
        assert output_dir.exists()
        assert list(output_dir.iterdir()) == []

        cm_result = ConfusionMatrixResult(
            tn=5, fp=1, fn=2, tp=8, matrix=np.array([[5, 1], [2, 8]])
        )

        try:
            saved_path = visualizer.plot_confusion_matrix(
                cm_result, model_name="unit-test-model"
            )
            assert Path(saved_path).exists()
        finally:
            plt.close("all")


class TestNoImportTimeSideEffects:
    def test_importing_evaluation_creates_no_files_or_directories(self):
        import utils.evaluation as evaluation

        results_existed_before = evaluation.RESULTS_DIR.exists()
        plots_existed_before = evaluation.PLOTS_DIR.exists()
        logs_existed_before = evaluation.LOGS_DIR.exists()

        importlib.reload(evaluation)

        if not results_existed_before:
            assert not evaluation.RESULTS_DIR.exists()
        if not plots_existed_before:
            assert not evaluation.PLOTS_DIR.exists()
        if not logs_existed_before:
            assert not evaluation.LOGS_DIR.exists()
