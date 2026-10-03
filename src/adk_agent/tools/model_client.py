"""HTTP client for the model inference server.

All tool functions call this instead of loading models in-process,
avoiding torch segfaults under Streamlit's process model.

Configure the server URL via MODEL_SERVER_URL env var (default: http://localhost:8100).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
from typing import Any, Dict, Iterator

import httpx

_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8100")
_TIMEOUT = 120.0
_COST_CONTEXT: ContextVar[Dict[str, str]] = ContextVar("adk_model_cost_context", default={})


@contextmanager
def model_cost_context(story: str | None = None, workflow_step: str | None = None) -> Iterator[None]:
    """Attach optional cost-evidence context to model-server HTTP calls."""
    context = {
        "story": story or "",
        "workflow_step": workflow_step or "",
    }
    token = _COST_CONTEXT.set(context)
    try:
        yield
    finally:
        _COST_CONTEXT.reset(token)


def _cost_headers() -> Dict[str, str]:
    context = _COST_CONTEXT.get({})
    headers: Dict[str, str] = {}
    if context.get("story"):
        headers["X-ADK-Cost-Story"] = context["story"]
    if context.get("workflow_step"):
        headers["X-ADK-Cost-Workflow-Step"] = context["workflow_step"]
    return headers


def predict(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to the model server and return the JSON response."""
    url = f"{_SERVER_URL}{endpoint}"
    resp = httpx.post(url, json=payload, headers=_cost_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def server_healthy() -> bool:
    """Quick check that the model server is reachable."""
    try:
        resp = httpx.get(f"{_SERVER_URL}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False
