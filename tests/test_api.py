"""Tests for api/app.py.

The inference boundary (health_check, predict_news) is monkeypatched at the
api.app module level for every test, so no real model is ever loaded.
"""

import pytest
from fastapi.testclient import TestClient

import api.app as app_module
from api.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _status(
    status="healthy",
    xgboost_available=True,
    bert_artifact_available=False,
    bert_loaded=False,
    bert_load_error=None,
):
    return {
        "status": status,
        "xgboost_available": xgboost_available,
        "bert_artifact_available": bert_artifact_available,
        "bert_loaded": bert_loaded,
        "bert_load_error": bert_load_error,
    }


class TestRootEndpoint:
    def test_healthy_root_response(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "health_check", lambda: _status())

        response = client.get("/")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["project"] == "Fatocheck - Fake News Detection API"
        assert body["version"] == "2.0.0"
        assert body["xgboost_ready"] is True
        assert "xgboost" in body["available_models"]
        assert "bert" not in body["available_models"]

    def test_available_models_reporting_includes_bert(self, client, monkeypatch):
        monkeypatch.setattr(
            app_module, "health_check", lambda: _status(bert_artifact_available=True)
        )

        response = client.get("/")

        assert set(response.json()["available_models"]) == {"xgboost", "bert"}


class TestHealthEndpoint:
    def test_healthy(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "health_check", lambda: _status())

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_degraded_model_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(
            app_module,
            "health_check",
            lambda: _status(status="degraded", xgboost_available=False),
        )

        response = client.get("/health")

        assert response.status_code == 503


class TestReadyEndpoint:
    def test_successful_readiness(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "health_check", lambda: _status())

        response = client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_readiness_returns_503_when_xgboost_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(
            app_module, "health_check", lambda: _status(xgboost_available=False)
        )

        response = client.get("/ready")

        assert response.status_code == 503


class TestModelsEndpoint:
    @pytest.mark.parametrize(
        "bert_loaded,bert_artifact_available,expected_status",
        [
            (True, True, "loaded"),
            (False, True, "lazy-loaded"),
            (False, False, "unavailable"),
        ],
    )
    def test_bert_metadata_states(
        self, client, monkeypatch, bert_loaded, bert_artifact_available, expected_status
    ):
        monkeypatch.setattr(
            app_module,
            "health_check",
            lambda: _status(
                bert_loaded=bert_loaded,
                bert_artifact_available=bert_artifact_available,
            ),
        )

        response = client.get("/models")

        assert response.json()["models"]["bert"]["status"] == expected_status

    def test_xgboost_metadata_unavailable_state(self, client, monkeypatch):
        monkeypatch.setattr(
            app_module, "health_check", lambda: _status(xgboost_available=False)
        )

        response = client.get("/models")

        assert response.json()["models"]["xgboost"]["status"] == "unavailable"


class TestPredictEndpoint:
    def test_successful_xgboost_prediction(
        self, client, monkeypatch, sample_prediction_result
    ):
        monkeypatch.setattr(
            app_module, "predict_news", lambda text, model_type: sample_prediction_result
        )

        response = client.post(
            "/predict", json={"text": "Some news text", "model": "xgboost"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["result"] == sample_prediction_result

    def test_successful_bert_prediction(self, client, monkeypatch):
        expected = {
            "model": "fatocheck-bert",
            "prediction": "Fake",
            "confidence": 0.6,
            "processing_time_seconds": 0.05,
        }
        monkeypatch.setattr(app_module, "predict_news", lambda text, model_type: expected)

        response = client.post(
            "/predict", json={"text": "Some news text", "model": "bert"}
        )

        assert response.status_code == 200
        assert response.json()["result"] == expected

    def test_whitespace_only_text_returns_400(self, client):
        response = client.post("/predict", json={"title": "  ", "text": " "})

        assert response.status_code == 400

    def test_empty_text_rejected_by_schema(self, client):
        response = client.post("/predict", json={"text": ""})

        assert response.status_code == 422

    def test_invalid_model_name_rejected_by_schema(self, client):
        response = client.post(
            "/predict", json={"text": "Some text", "model": "not-a-model"}
        )

        assert response.status_code == 422

    def test_value_error_mapped_to_400(self, client, monkeypatch):
        def fake_predict_news(text, model_type):
            raise ValueError("bad input")

        monkeypatch.setattr(app_module, "predict_news", fake_predict_news)

        response = client.post("/predict", json={"text": "Some text"})

        assert response.status_code == 400
        assert response.json()["detail"] == "bad input"

    def test_missing_artifact_mapped_to_503(self, client, monkeypatch):
        def fake_predict_news(text, model_type):
            raise FileNotFoundError("artifact missing")

        monkeypatch.setattr(app_module, "predict_news", fake_predict_news)

        response = client.post("/predict", json={"text": "Some text"})

        assert response.status_code == 503

    def test_runtime_failure_mapped_to_503(self, client, monkeypatch):
        def fake_predict_news(text, model_type):
            raise RuntimeError("model broke")

        monkeypatch.setattr(app_module, "predict_news", fake_predict_news)

        response = client.post("/predict", json={"text": "Some text"})

        assert response.status_code == 503

    def test_unexpected_exception_mapped_to_500(self, client, monkeypatch):
        def fake_predict_news(text, model_type):
            raise KeyError("boom")

        monkeypatch.setattr(app_module, "predict_news", fake_predict_news)

        response = client.post("/predict", json={"text": "Some text"})

        assert response.status_code == 500
        assert response.json()["detail"] == "Internal error while running inference."
