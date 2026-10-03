"""Cost evidence helpers for local model serving.

The project serves fine-tuned local artefacts through FastAPI. These helpers
estimate what the same specialist inference payload would cost if sent to a
Gemini 3.1 Flash-Lite style API. The numbers are intended for academic
comparison, not billing reconciliation.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable


GEMINI_EQUIVALENT_MODEL = os.getenv(
    "GEMINI_COST_BASIS_MODEL",
    "gemini-3.1-flash-lite",
)
GEMINI_INPUT_USD_PER_1M = float(
    os.getenv("GEMINI_FLASH_LITE_INPUT_USD_PER_1M", "0.10")
)
GEMINI_OUTPUT_USD_PER_1M = float(
    os.getenv("GEMINI_FLASH_LITE_OUTPUT_USD_PER_1M", "0.40")
)
PRICING_BASIS = os.getenv(
    "GEMINI_COST_PRICING_BASIS",
    "Gemini API listed price for Gemini 3.1 Flash-Lite, text/image/video tier",
)


ENDPOINT_MODEL_MAP: Dict[str, tuple[str, str]] = {
    "/predict/sentiment": ("distilbert-sentiment", "local_classifier"),
    "/predict/sentiment/batch": ("distilbert-sentiment", "local_classifier"),
    "/predict/churn_xgboost": ("xgboost-churn", "local_classifier"),
    "/predict/churn_llm": ("qwen2.5-0.5b-lora-churn", "local_slm"),
    "/predict/segmentation": ("qwen2.5-0.5b-lora-segmentation", "local_slm"),
    "/predict/segmentation/batch": ("qwen2.5-0.5b-lora-segmentation", "local_slm"),
    "/predict/support": ("qwen2.5-0.5b-lora-support", "local_slm"),
    "/predict/content": ("qwen2.5-0.5b-lora-content", "local_slm"),
    "/predict/recommendation": ("qwen2.5-0.5b-lora-recommendation", "local_slm"),
    "/predict/recommendation/batch": ("qwen2.5-0.5b-lora-recommendation", "local_slm"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def estimate_tokens(value: Any) -> int:
    """Estimate token count using a conservative character-based proxy."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = text.strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def gemini_equivalent_cost_usd(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * GEMINI_INPUT_USD_PER_1M
    output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_USD_PER_1M
    return round(input_cost + output_cost, 8)


def endpoint_model(endpoint: str) -> tuple[str, str]:
    return ENDPOINT_MODEL_MAP.get(endpoint, ("unknown-local-model", "local_model"))


def build_model_cost_event(
    *,
    endpoint: str,
    payload: Dict[str, Any],
    response: Any,
    latency_ms: float,
    story: str | None = None,
    workflow_step: str | None = None,
) -> Dict[str, Any]:
    input_tokens = estimate_tokens(payload)
    output_tokens = estimate_tokens(response)
    total_tokens = input_tokens + output_tokens
    model_name, provider_type = endpoint_model(endpoint)
    equivalent_cost = gemini_equivalent_cost_usd(input_tokens, output_tokens)
    return {
        "timestamp": utc_now(),
        "story": story or "unknown",
        "workflow_step": workflow_step or "",
        "endpoint": endpoint,
        "model_name": model_name,
        "provider_type": provider_type,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_ms": round(latency_ms, 1),
        "pricing_basis_model": GEMINI_EQUIVALENT_MODEL,
        "pricing_basis": PRICING_BASIS,
        "gemini_input_usd_per_1m": GEMINI_INPUT_USD_PER_1M,
        "gemini_output_usd_per_1m": GEMINI_OUTPUT_USD_PER_1M,
        "gemini_equivalent_cost_usd": equivalent_cost,
        "actual_api_cost_usd": 0.0,
    }


def gemini_usage_from_monitoring_events(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    calls = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cached_tokens = 0
    cost = 0.0
    for event in events:
        if event.get("event_type") != "model_call":
            continue
        calls += 1
        prompt = int(event.get("prompt_tokens") or 0)
        completion = int(event.get("completion_tokens") or 0)
        input_tokens += prompt
        output_tokens += completion
        total_tokens += int(event.get("total_tokens") or prompt + completion)
        cached_tokens += int(event.get("cached_tokens") or 0)
        cost += gemini_equivalent_cost_usd(prompt, completion)
    return {
        "gemini_model_calls": calls,
        "gemini_input_tokens": input_tokens,
        "gemini_output_tokens": output_tokens,
        "gemini_total_tokens": total_tokens,
        "gemini_cached_tokens": cached_tokens,
        "gemini_token_cost_usd": round(cost, 8),
    }


def gemini_cost_from_monitoring_events(events: Iterable[Dict[str, Any]]) -> float:
    return float(gemini_usage_from_monitoring_events(events)["gemini_token_cost_usd"])


def summarize_cost_evidence(
    cost_events: Iterable[Dict[str, Any]],
    monitoring_events: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    rows = list(cost_events)
    local_input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows)
    local_output_tokens = sum(int(row.get("output_tokens") or 0) for row in rows)
    local_total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    local_equivalent_cost = gemini_equivalent_cost_usd(
        local_input_tokens,
        local_output_tokens,
    )
    gemini_usage = (
        gemini_usage_from_monitoring_events(monitoring_events or [])
        if monitoring_events is not None
        else {
            "gemini_model_calls": 0,
            "gemini_input_tokens": 0,
            "gemini_output_tokens": 0,
            "gemini_total_tokens": 0,
            "gemini_cached_tokens": 0,
            "gemini_token_cost_usd": 0.0,
        }
    )
    gemini_orchestration_cost = float(gemini_usage["gemini_token_cost_usd"])
    all_gemini_baseline_cost = round(
        gemini_orchestration_cost + local_equivalent_cost,
        8,
    )
    cost_reduction = local_equivalent_cost
    reduction_pct = (
        cost_reduction / all_gemini_baseline_cost * 100
        if all_gemini_baseline_cost > 0
        else 0.0
    )
    return {
        **gemini_usage,
        "local_model_calls": len(rows),
        "local_input_tokens": local_input_tokens,
        "local_output_tokens": local_output_tokens,
        "local_total_tokens": local_total_tokens,
        "local_gemini_equivalent_cost_usd": local_equivalent_cost,
        "total_gemini_token_cost_usd": gemini_orchestration_cost,
        "actual_gemini_orchestration_cost_usd": gemini_orchestration_cost,
        "all_gemini_baseline_cost_usd": all_gemini_baseline_cost,
        "estimated_api_cost_avoided_usd": cost_reduction,
        "estimated_api_cost_reduction_usd": cost_reduction,
        "estimated_api_cost_reduction_pct": round(reduction_pct, 2),
        "pricing_basis_model": GEMINI_EQUIVALENT_MODEL,
        "pricing_basis": PRICING_BASIS,
    }
