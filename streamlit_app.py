"""
FatoCheck Streamlit UI

Presentation layer only. All predictions go through the FastAPI backend
over HTTP. This module does not import inference, preprocessing, or models.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

from __future__ import annotations

import json
import os
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


def get_api_base_url() -> str:
    """Return the FastAPI base URL from the environment."""
    return os.getenv("FATOCHECK_API_URL", DEFAULT_API_URL).rstrip("/")


# ------------------------------------------------------------------
# HTTP Client
# ------------------------------------------------------------------


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any] | None, dict[str, str]]:
    """Send an HTTP request and return status, JSON body, and response headers."""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_id = str(uuid.uuid4())
    headers = {
        "Accept": "application/json",
        REQUEST_ID_HEADER: request_id,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"

    request = Request(url=url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw) if raw else None
            response_headers = {key: value for key, value in response.headers.items()}
            return response.status, parsed, response_headers
    except HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp is not None else ""
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = None
        response_headers = {key: value for key, value in exc.headers.items()} if exc.headers else {}
        return exc.code, parsed, response_headers
    except URLError as exc:
        raise ConnectionError(f"Unable to reach the FatoCheck API at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise TimeoutError(f"The FatoCheck API timed out after {timeout}s.") from exc


# ------------------------------------------------------------------
# Backend API Functions
# ------------------------------------------------------------------


def fetch_health(api_base: str) -> dict[str, Any]:
    status, body, _ = _http_json("GET", f"{api_base}/health")
    if status == 200 and isinstance(body, dict):
        return {"ok": True, "body": body}
    message = _extract_error_message(body) or "API health check failed."
    return {"ok": False, "status_code": status, "message": message, "body": body}


def fetch_models(api_base: str) -> dict[str, Any] | None:
    status, body, _ = _http_json("GET", f"{api_base}/models")
    if status == 200 and isinstance(body, dict):
        return body
    return None


def predict_article(
    api_base: str,
    *,
    title: str,
    text: str,
    model: str,
) -> tuple[bool, dict[str, Any], dict[str, str]]:
    status, body, headers = _http_json(
        "POST",
        f"{api_base}/predict",
        payload={"title": title, "text": text, "model": model},
    )
    if status == 200 and isinstance(body, dict) and body.get("success") is True:
        return True, body, headers
    if not isinstance(body, dict):
        body = {
            "success": False,
            "error": {
                "code": "HTTP_ERROR",
                "message": f"Unexpected API response (HTTP {status}).",
                "request_id": headers.get(REQUEST_ID_HEADER, ""),
            },
        }
    return False, body, headers


# ------------------------------------------------------------------
# Response Parsing
# ------------------------------------------------------------------


def _extract_error_message(body: dict[str, Any] | None) -> str | None:
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    detail = body.get("detail")
    if isinstance(detail, str):
        return detail
    return None


# ------------------------------------------------------------------
# UI Rendering Helpers
# ------------------------------------------------------------------


def render_health_indicator(api_base: str) -> None:
    try:
        health = fetch_health(api_base)
    except (ConnectionError, TimeoutError) as exc:
        st.error(f"Backend unavailable: {exc}")
        return

    if health["ok"]:
        st.success("API status: healthy")
        body = health["body"]
        bert_note = "available" if body.get("bert_artifact_available") else "not available in this deployment"
        st.caption(
            f"XGBoost production model ready. Local BERT artifact: {bert_note}. "
            f"BERT loaded in memory: {body.get('bert_loaded', False)}."
        )
    else:
        st.warning(f"API status: degraded/unavailable — {health.get('message')}")


def render_prediction_result(body: dict[str, Any], headers: dict[str, str]) -> None:
    result = body.get("result", {})
    prediction = str(result.get("prediction", "Unknown"))
    confidence = result.get("confidence")
    processing_time = result.get("processing_time_seconds")
    model_name = result.get("model", "unknown")

    st.subheader("Prediction")
    if prediction.lower() == "real":
        st.success(f"Label: **{prediction}**")
    elif prediction.lower() == "fake":
        st.error(f"Label: **{prediction}**")
    else:
        st.info(f"Label: **{prediction}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        if isinstance(confidence, (int, float)):
            st.metric("Confidence", f"{confidence:.2%}" if confidence <= 1 else f"{confidence:.4f}")
        else:
            st.metric("Confidence", "n/a")
    with col2:
        if isinstance(processing_time, (int, float)):
            st.metric("Processing time", f"{processing_time:.4f}s")
        else:
            st.metric("Processing time", "n/a")
    with col3:
        st.metric("Model", model_name)

    with st.expander("Technical details"):
        request_id = headers.get(REQUEST_ID_HEADER) or (body.get("error", {}) or {}).get("request_id", "n/a")
        process_time_header = headers.get(PROCESS_TIME_HEADER, "n/a")
        st.code(
            f"request_id: {request_id}\n" f"X-Process-Time: {process_time_header}\n" f"model: {model_name}",
            language="text",
        )


def render_error(body: dict[str, Any], headers: dict[str, str]) -> None:
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        code = error.get("code", "ERROR")
        message = error.get("message", "Request failed.")
        request_id = error.get("request_id") or headers.get(REQUEST_ID_HEADER, "n/a")
        st.error(f"{code}: {message}")
        with st.expander("Technical details"):
            st.code(f"request_id: {request_id}", language="text")
        return

    st.error(_extract_error_message(body) or "Request failed.")


# ------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # Page Configuration
    # ------------------------------------------------------------------
    st.set_page_config(
        page_title="FatoCheck",
        layout="centered",
    )

    api_base = get_api_base_url()
    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("FatoCheck")
    st.write(
        "AI-powered fake-news classification demo. "
        "This UI sends requests to the FastAPI backend; "
        "models are never loaded inside Streamlit."
    )
    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    with st.expander("Architecture", expanded=False):
        st.code(
            "Browser\n"
            "   │\n"
            "   ▼\n"
            "Streamlit UI\n"
            "   │  HTTP\n"
            "   ▼\n"
            "FastAPI\n"
            "   │\n"
            "Inference Layer\n"
            "   ├── XGBoost (production)\n"
            "   └── BERT (optional / local)",
            language="text",
        )
        st.caption(f"API base URL: `{api_base}` (override with `FATOCHECK_API_URL`)")

    # ------------------------------------------------------------------
    # Backend Status
    # ------------------------------------------------------------------
    st.subheader("Backend status")
    render_health_indicator(api_base)
    # ------------------------------------------------------------------
    # Model Configuration
    # ------------------------------------------------------------------
    model_options = ["xgboost", "bert"]
    models_payload = None
    try:
        models_payload = fetch_models(api_base)
    except (ConnectionError, TimeoutError):
        models_payload = None

    model_help = "XGBoost is the production model. BERT requires a complete local artifact on the API host."
    if models_payload and isinstance(models_payload.get("models"), dict):
        xgb_status = models_payload["models"].get("xgboost", {}).get("status", "unknown")
        bert_status = models_payload["models"].get("bert", {}).get("status", "unknown")
        model_help = f"API reports XGBoost as `{xgb_status}` and BERT as `{bert_status}`."
    # ------------------------------------------------------------------
    # Article Analysis Form
    # ------------------------------------------------------------------
    st.subheader("Analyze an article")
    model = st.selectbox("Model", options=model_options, index=0, help=model_help)
    title = st.text_input("Headline (optional)", placeholder="Optional news headline")
    text = st.text_area(
        "Article text",
        height=220,
        placeholder="Paste the news article body here.",
    )
    # ------------------------------------------------------------------
    # Prediction Request
    # ------------------------------------------------------------------
    if st.button("Analyze", type="primary"):
        cleaned_title = (title or "").strip()
        cleaned_text = (text or "").strip()

        if not cleaned_text and not cleaned_title:
            st.warning("Please provide article text (headline alone is not enough).")
            return
        if not cleaned_text:
            st.warning("Article text cannot be empty.")
            return

        with st.spinner("Contacting the FatoCheck API..."):
            try:
                ok, body, headers = predict_article(
                    api_base,
                    title=cleaned_title,
                    text=cleaned_text,
                    model=model,
                )
            except ConnectionError as exc:
                st.error(str(exc))
                st.info("Start the API with: `uvicorn api.app:app --reload --port 8000`")
                return
            except TimeoutError as exc:
                st.error(str(exc))
                return

        if ok:
            render_prediction_result(body, headers)
        else:
            render_error(body, headers)

    # ------------------------------------------------------------------
    # Sidebar - Author Information
    # ------------------------------------------------------------------

    st.sidebar.divider()

    st.sidebar.markdown("### About")

    st.sidebar.markdown("**Susanta Hazra**")

    st.sidebar.caption("AI Engineer")

    st.sidebar.markdown(
        """
🔗 [LinkedIn](https://www.linkedin.com/in/susantahazra/)

💻 [GitHub](https://github.com/Susanta2025-lab)
"""
    )

    st.sidebar.divider()

    st.sidebar.caption("Powered by FastAPI • Streamlit • XGBoost • BERT")


# ------------------------------------------------------------------
# Application Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
