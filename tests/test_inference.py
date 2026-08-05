"""Tests for utils/inference.py.

The real XGBoost artifact is loaded at import time by this module, but every
test below overrides that global state with lightweight fakes so no real
model is exercised. The BERT model is never actually loaded.
"""

import pytest

import utils.inference as inference


class TestPredictNewsRouting:
    def test_empty_text_raises_value_error(self):
        with pytest.raises(ValueError):
            inference.predict_news("", model_type="xgboost")

    def test_whitespace_only_text_raises_value_error(self):
        with pytest.raises(ValueError):
            inference.predict_news("   ", model_type="xgboost")

    def test_unsupported_model_raises_value_error(self, sample_text):
        with pytest.raises(ValueError):
            inference.predict_news(sample_text, model_type="not-a-model")

    def test_xgboost_route_calls_xgboost_predictor(self, monkeypatch, sample_text):
        called = {}

        def fake_predict_xgboost(text):
            called["text"] = text
            return {
                "model": "xgboost",
                "prediction": "Real",
                "confidence": 0.9,
                "processing_time_seconds": 0.0,
            }

        monkeypatch.setattr(inference, "predict_xgboost", fake_predict_xgboost)

        result = inference.predict_news(sample_text, model_type="xgboost")

        assert called["text"] == sample_text
        assert result["model"] == "xgboost"

    def test_bert_route_calls_bert_predictor(self, monkeypatch, sample_text):
        called = {}

        def fake_predict_bert(text):
            called["text"] = text
            return {
                "model": "fatocheck-bert",
                "prediction": "Fake",
                "confidence": 0.6,
                "processing_time_seconds": 0.0,
            }

        monkeypatch.setattr(inference, "predict_bert", fake_predict_bert)

        result = inference.predict_news(sample_text, model_type="bert")

        assert called["text"] == sample_text
        assert result["model"] == "fatocheck-bert"


class TestPredictXGBoost:
    def test_response_contains_expected_fields(self, monkeypatch, mock_xgboost_pipeline):
        monkeypatch.setattr(inference, "xgboost_model", mock_xgboost_pipeline)

        result = inference.predict_xgboost("Some article text")

        assert set(result.keys()) == {
            "model",
            "prediction",
            "confidence",
            "processing_time_seconds",
        }
        assert result["model"] == "xgboost"

    def test_confidence_matches_predicted_class_real(
        self, monkeypatch, make_mock_xgboost_pipeline
    ):
        pipeline = make_mock_xgboost_pipeline(
            predicted_class=1, probabilities=(0.13, 0.87), classes=(0, 1)
        )
        monkeypatch.setattr(inference, "xgboost_model", pipeline)

        result = inference.predict_xgboost("Some article text")

        assert result["prediction"] == "Real"
        assert result["confidence"] == pytest.approx(0.87)

    def test_confidence_matches_predicted_class_fake(
        self, monkeypatch, make_mock_xgboost_pipeline
    ):
        pipeline = make_mock_xgboost_pipeline(
            predicted_class=0, probabilities=(0.91, 0.09), classes=(0, 1)
        )
        monkeypatch.setattr(inference, "xgboost_model", pipeline)

        result = inference.predict_xgboost("Some article text")

        assert result["prediction"] == "Fake"
        assert result["confidence"] == pytest.approx(0.91)

    def test_missing_model_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(inference, "xgboost_model", None)

        with pytest.raises(RuntimeError):
            inference.predict_xgboost("Some article text")


class TestBertArtifactChecks:
    def test_artifact_available_when_complete(self, monkeypatch, make_bert_artifact_dir):
        bert_dir = make_bert_artifact_dir()
        monkeypatch.setattr(inference, "BERT_MODEL_PATH", bert_dir)

        assert inference.is_bert_artifact_available() is True

    def test_artifact_unavailable_without_weight_file(
        self, monkeypatch, make_bert_artifact_dir
    ):
        bert_dir = make_bert_artifact_dir(weight_file=None)
        monkeypatch.setattr(inference, "BERT_MODEL_PATH", bert_dir)

        assert inference.is_bert_artifact_available() is False

    def test_artifact_unavailable_when_directory_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(inference, "BERT_MODEL_PATH", tmp_path / "does-not-exist")

        assert inference.is_bert_artifact_available() is False

    def test_incomplete_artifact_reports_missing_files_clearly(
        self, monkeypatch, make_bert_artifact_dir
    ):
        bert_dir = make_bert_artifact_dir(include_tokenizer=False)
        monkeypatch.setattr(inference, "BERT_MODEL_PATH", bert_dir)

        with pytest.raises(FileNotFoundError) as exc_info:
            inference._validate_bert_artifact()

        assert "tokenizer.json" in str(exc_info.value)

    def test_validate_missing_directory_raises_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(inference, "BERT_MODEL_PATH", tmp_path / "missing-dir")

        with pytest.raises(FileNotFoundError) as exc_info:
            inference._validate_bert_artifact()

        assert "not found" in str(exc_info.value).lower()


class TestPredictBert:
    def test_predict_bert_uses_mocked_model(self, monkeypatch, sample_text):
        import torch

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }

        class FakeConfig:
            id2label = {0: "Fake", 1: "Real"}
            num_labels = 2

        class FakeOutput:
            def __init__(self, logits):
                self.logits = logits

        class FakeModel:
            config = FakeConfig()

            def __call__(self, **inputs):
                return FakeOutput(torch.tensor([[0.1, 2.5]]))

        monkeypatch.setattr(
            inference, "load_bert_model", lambda: (FakeTokenizer(), FakeModel())
        )

        result = inference.predict_bert(sample_text)

        assert result["model"] == "fatocheck-bert"
        assert result["prediction"] == "Real"
        assert 0.0 <= result["confidence"] <= 1.0
        assert "processing_time_seconds" in result


class TestHealthCheck:
    def test_reflects_xgboost_availability_true(self, monkeypatch, mock_xgboost_pipeline):
        monkeypatch.setattr(inference, "xgboost_model", mock_xgboost_pipeline)
        monkeypatch.setattr(inference, "is_bert_artifact_available", lambda: False)

        status = inference.health_check()

        assert status["xgboost_available"] is True
        assert status["status"] == "healthy"

    def test_reflects_xgboost_availability_false(self, monkeypatch):
        monkeypatch.setattr(inference, "xgboost_model", None)
        monkeypatch.setattr(inference, "is_bert_artifact_available", lambda: False)

        status = inference.health_check()

        assert status["xgboost_available"] is False
        assert status["status"] == "degraded"

    def test_does_not_trigger_bert_loading(self, monkeypatch, mock_xgboost_pipeline):
        def boom():
            raise AssertionError("health_check must not trigger BERT loading")

        monkeypatch.setattr(inference, "xgboost_model", mock_xgboost_pipeline)
        monkeypatch.setattr(inference, "load_bert_model", boom)

        status = inference.health_check()  # Must not raise.

        assert status["bert_loaded"] is False

    def test_bert_lazy_load_state_reflected(self, monkeypatch, mock_xgboost_pipeline):
        monkeypatch.setattr(inference, "xgboost_model", mock_xgboost_pipeline)
        monkeypatch.setattr(inference, "_bert_model", object())

        status = inference.health_check()

        assert status["bert_loaded"] is True


@pytest.mark.integration
def test_real_xgboost_artifact_predicts():
    """Optional integration check for the real XGBoost artifact.

    Skipped automatically when the artifact is unavailable or cannot be
    loaded (e.g. due to a library version mismatch); never required for
    normal test runs.
    """
    if not inference.XGBOOST_MODEL_PATH.exists():
        pytest.skip("Real XGBoost artifact not available")

    import joblib

    try:
        real_model = joblib.load(inference.XGBOOST_MODEL_PATH)
        prediction = real_model.predict(
            ["A neutral sample sentence about a routine economic report."]
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load/run real XGBoost artifact: {exc}")

    assert int(prediction[0]) in (0, 1)
