"""Shared pytest fixtures for the FatoCheck test suite.

Only fixtures that are genuinely reused across multiple test modules live
here; test-specific helpers stay local to their test file.
"""

import matplotlib

matplotlib.use("Agg")  # Must be set before any pyplot import (utils.evaluation).

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def sample_text():
    """A representative raw news snippet exercising several cleaning rules."""
    return "Check this OUT: http://example.com <b>BREAKING</b> " "News!!! Call 911 now.   Multiple   spaces."


@pytest.fixture
def sample_dataframe():
    """A tiny, fully in-memory DataFrame shaped like the WELFake dataset."""
    return pd.DataFrame(
        {
            "title": ["Breaking News", None, "Another Title"],
            "text": ["Some article text.", "Second article text.", None],
            "label": [1, 0, 1],
        }
    )


class FakeXGBoostPipeline:
    """Minimal stand-in for the real TF-IDF + XGBoost sklearn Pipeline.

    Only implements the interface predict_xgboost() actually relies on:
    predict(), predict_proba(), and classes_.
    """

    def __init__(self, predicted_class, probabilities, classes=(0, 1)):
        self.classes_ = list(classes)
        self._predicted_class = predicted_class
        self._probabilities = probabilities

    def predict(self, X):
        return [self._predicted_class for _ in X]

    def predict_proba(self, X):
        return [self._probabilities for _ in X]


@pytest.fixture
def make_mock_xgboost_pipeline():
    """Factory for building fake XGBoost pipelines with a chosen outcome."""

    def _factory(predicted_class=1, probabilities=(0.13, 0.87), classes=(0, 1)):
        return FakeXGBoostPipeline(predicted_class, probabilities, classes)

    return _factory


@pytest.fixture
def mock_xgboost_pipeline(make_mock_xgboost_pipeline):
    """A default fake XGBoost pipeline predicting REAL with high confidence."""
    return make_mock_xgboost_pipeline()


@pytest.fixture
def sample_prediction_result():
    """A representative payload shaped like predict_xgboost()'s output."""
    return {
        "model": "xgboost",
        "prediction": "Real",
        "confidence": 0.87,
        "processing_time_seconds": 0.001,
    }


@pytest.fixture
def make_bert_artifact_dir(tmp_path):
    """Factory that builds a synthetic local BERT artifact directory."""

    def _factory(
        include_config=True,
        include_tokenizer_config=True,
        include_tokenizer=True,
        weight_file="model.safetensors",
    ):
        bert_dir = tmp_path / "bert-base-uncased"
        bert_dir.mkdir(parents=True, exist_ok=True)

        if include_config:
            (bert_dir / "config.json").write_text("{}")
        if include_tokenizer_config:
            (bert_dir / "tokenizer_config.json").write_text("{}")
        if include_tokenizer:
            (bert_dir / "tokenizer.json").write_text("{}")
        if weight_file:
            (bert_dir / weight_file).write_bytes(b"fake-weights")

        return bert_dir

    return _factory
