"""FastAPI model inference server — runs models in a separate process.

Torch model loading segfaults under uvloop (used by both Streamlit and
uvicorn).  This server loads all models synchronously before uvicorn
starts, then serves predictions via HTTP on the standard asyncio loop.

Start the server first, then launch the Streamlit demo:

    # Terminal 1 — model server
    python -m src.adk_agent.model_server

    # Terminal 2 — Streamlit demo
    streamlit run src/adk_agent/demo.py
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# XGBoost's OpenMP thread pool deadlocks with safetensors' parallel loading
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adk_agent.tools.model_loader import (
    get_churn_llm_model,
    get_churn_xgboost_model,
    get_content_model,
    get_recommendation_model,
    get_segmentation_model,
    get_sentiment_model,
    get_support_model,
)
from src.adk_agent.tools.churn_tools import _encode_and_scale, _engineer_features
from src.adk_agent.cost_tracking import build_model_cost_event
from src.adk_agent.monitoring_store import save_model_cost_event

logger = logging.getLogger(__name__)

# ── Request schemas ──────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

class ChurnXGBoostRequest(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    internet_service: str
    payment_method: str
    gender: str = "Male"
    senior_citizen: int = 0
    partner: str = "No"
    dependents: str = "No"
    phone_service: str = "Yes"
    multiple_lines: str = "No"
    online_security: str = "No"
    online_backup: str = "No"
    device_protection: str = "No"
    tech_support: str = "No"
    streaming_tv: str = "No"
    streaming_movies: str = "No"
    paperless_billing: str = "Yes"


# ── Model cache ──────────────────────────────────────────────────────

_models: Dict[str, Any] = {}


def _preload_models():
    """Load all models synchronously — called before uvicorn starts."""
    t0 = time.time()
    logger.info("Loading 7 models (5 LoRA models need ~30s each) …")

    _models["churn_xgb"] = get_churn_xgboost_model()
    logger.info("[1/7] XGBoost churn loaded")

    _models["sentiment"] = get_sentiment_model()
    logger.info("[2/7] Sentiment (DistilBERT) loaded")

    _models["churn_llm"] = get_churn_llm_model()
    logger.info("[3/7] Churn LLM (LoRA) loaded")

    _models["segmentation"] = get_segmentation_model()
    logger.info("[4/7] Segmentation LLM loaded")

    _models["support"] = get_support_model()
    logger.info("[5/7] Support LLM loaded")

    _models["content"] = get_content_model()
    logger.info("[6/7] Content LLM loaded")

    _models["recommendation"] = get_recommendation_model()
    logger.info("[7/7] Recommendation LLM loaded")

    elapsed = time.time() - t0
    logger.info("All 7 models loaded in %.1fs — server ready", elapsed)


app = FastAPI(title="AI Marketing Model Server")


def _cost_context(request: Request) -> tuple[str | None, str | None]:
    return (
        request.headers.get("X-ADK-Cost-Story"),
        request.headers.get("X-ADK-Cost-Workflow-Step"),
    )


def _tracked_prediction(
    request: Request,
    endpoint: str,
    payload: Dict[str, Any],
    predict_fn,
) -> Any:
    start = time.time()
    response = predict_fn()
    latency_ms = (time.time() - start) * 1000
    story, workflow_step = _cost_context(request)
    try:
        save_model_cost_event(
            build_model_cost_event(
                endpoint=endpoint,
                payload=payload,
                response=response,
                latency_ms=latency_ms,
                story=story,
                workflow_step=workflow_step,
            )
        )
    except Exception as exc:
        logger.debug("Could not persist model cost event: %s", exc)
    return response


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


# ── Sentiment ────────────────────────────────────────────────────────

@app.post("/predict/sentiment")
def predict_sentiment(req: TextRequest, request: Request):
    payload = {"text": req.text}
    return _tracked_prediction(
        request,
        "/predict/sentiment",
        payload,
        lambda: _models["sentiment"].predict(req.text),
    )


@app.post("/predict/sentiment/batch")
def predict_sentiment_batch(req: BatchTextRequest, request: Request):
    payload = {"texts": req.texts}
    return _tracked_prediction(
        request,
        "/predict/sentiment/batch",
        payload,
        lambda: _models["sentiment"].predict_batch(req.texts),
    )


# ── Churn (XGBoost) ─────────────────────────────────────────────────

@app.post("/predict/churn_xgboost")
def predict_churn_xgboost(req: ChurnXGBoostRequest, request: Request):
    raw = {
        "tenure": req.tenure,
        "MonthlyCharges": req.monthly_charges,
        "TotalCharges": req.total_charges,
        "Contract": req.contract,
        "InternetService": req.internet_service,
        "PaymentMethod": req.payment_method,
        "gender": req.gender,
        "SeniorCitizen": req.senior_citizen,
        "Partner": req.partner,
        "Dependents": req.dependents,
        "PhoneService": req.phone_service,
        "MultipleLines": req.multiple_lines,
        "OnlineSecurity": req.online_security,
        "OnlineBackup": req.online_backup,
        "DeviceProtection": req.device_protection,
        "TechSupport": req.tech_support,
        "StreamingTV": req.streaming_tv,
        "StreamingMovies": req.streaming_movies,
        "PaperlessBilling": req.paperless_billing,
    }
    def _predict():
        df = pd.DataFrame([raw])
        df = _engineer_features(df)
        predictor = _models["churn_xgb"]
        df = _encode_and_scale(df, predictor.label_encoders, predictor.scaler)
        prob = float(predictor.model.predict_proba(df[predictor.feature_names])[:, 1][0])
        label = "HIGH_RISK" if prob >= 0.5 else "LOW_RISK"
        return {"churn_probability": round(prob, 4), "risk": label}

    return _tracked_prediction(request, "/predict/churn_xgboost", raw, _predict)


# ── Churn (LLM) ─────────────────────────────────────────────────────

@app.post("/predict/churn_llm")
def predict_churn_llm(req: TextRequest, request: Request):
    payload = {"text": req.text}
    return _tracked_prediction(
        request,
        "/predict/churn_llm",
        payload,
        lambda: _models["churn_llm"].predict(req.text),
    )


# ── Segmentation ─────────────────────────────────────────────────────

@app.post("/predict/segmentation")
def predict_segmentation(req: TextRequest, request: Request):
    payload = {"text": req.text}
    return _tracked_prediction(
        request,
        "/predict/segmentation",
        payload,
        lambda: _models["segmentation"].predict(req.text),
    )


@app.post("/predict/segmentation/batch")
def predict_segmentation_batch(req: BatchTextRequest, request: Request):
    payload = {"texts": req.texts}
    return _tracked_prediction(
        request,
        "/predict/segmentation/batch",
        payload,
        lambda: _models["segmentation"].predict_batch(req.texts),
    )


# ── Support ──────────────────────────────────────────────────────────

@app.post("/predict/support")
def predict_support(req: TextRequest, request: Request):
    payload = {"text": req.text}
    return _tracked_prediction(
        request,
        "/predict/support",
        payload,
        lambda: _models["support"].predict(req.text),
    )


# ── Content ──────────────────────────────────────────────────────────

@app.post("/predict/content")
def predict_content(req: TextRequest, request: Request):
    payload = {"text": req.text}
    return _tracked_prediction(
        request,
        "/predict/content",
        payload,
        lambda: _models["content"].predict(req.text),
    )


# ── Recommendation ───────────────────────────────────────────────────

@app.post("/predict/recommendation")
def predict_recommendation(req: TextRequest, request: Request):
    payload = {"text": req.text}
    return _tracked_prediction(
        request,
        "/predict/recommendation",
        payload,
        lambda: _models["recommendation"].predict(req.text),
    )


@app.post("/predict/recommendation/batch")
def predict_recommendation_batch(req: BatchTextRequest, request: Request):
    payload = {"texts": req.texts}
    return _tracked_prediction(
        request,
        "/predict/recommendation/batch",
        payload,
        lambda: _models["recommendation"].predict_batch(req.texts),
    )


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Load all models synchronously BEFORE uvicorn/uvloop starts
    _preload_models()

    logger.info("Starting server on http://0.0.0.0:8100 (PID %d)", os.getpid())

    # Force standard asyncio loop (uvloop causes segfaults with torch)
    uvicorn.run(app, host="0.0.0.0", port=8100, loop="asyncio")
