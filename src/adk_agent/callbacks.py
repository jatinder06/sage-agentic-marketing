"""Monitoring and observability callbacks for the ADK multi-agent system.

Implements all 7 ADK callback hook points to track:
- Model call latency and token usage
- Tool execution timing and success rates
- Error rates and types
- Session-level metric accumulation
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.adk_agent.audit import log_audit_event
from src.adk_agent.monitoring_store import (
    clear_monitoring_events,
    load_monitoring_events,
    save_monitoring_event,
)

logger = logging.getLogger(__name__)

# Session state keys for metrics accumulation
_M = "_metrics"
_TIMER_PREFIX = "_timer."

# ---------------------------------------------------------------------------
# Per-event structured log (ring buffer for dashboard consumption)
# ---------------------------------------------------------------------------
_event_log: List[Dict[str, Any]] = []
_EVENT_LOG_MAX = 5000


def get_event_log() -> List[Dict[str, Any]]:
    """Return persisted monitoring events, falling back to memory."""
    try:
        return load_monitoring_events(limit=_EVENT_LOG_MAX)
    except Exception as exc:
        logger.debug("Could not load persisted monitoring events: %s", exc)
        return list(_event_log)


def clear_event_log() -> None:
    """Clear monitoring events from memory and SQLite."""
    _event_log.clear()
    try:
        clear_monitoring_events()
    except Exception as exc:
        logger.debug("Could not clear persisted monitoring events: %s", exc)


def _log_event(record: Dict[str, Any]) -> None:
    _event_log.append(record)
    if len(_event_log) > _EVENT_LOG_MAX:
        del _event_log[: len(_event_log) - _EVENT_LOG_MAX]
    try:
        save_monitoring_event(record)
    except Exception as exc:
        logger.debug("Could not persist monitoring event: %s", exc)


def _context_identity(callback_context: Any) -> Dict[str, Any]:
    state = getattr(callback_context, "state", {}) or {}
    return {
        "user_id": (
            state.get("adk.user_id")
            or getattr(callback_context, "user_id", None)
            or "anonymous"
        ),
        "role": state.get("adk.role", "unknown"),
        "session_id": (
            state.get("adk.session_id")
            or getattr(callback_context, "session_id", None)
            or "unknown"
        ),
    }


def _inc(state: Dict[str, Any], key: str, amount: int = 1) -> None:
    full = f"{_M}.{key}"
    state[full] = state.get(full, 0) + amount


def _append(state: Dict[str, Any], key: str, value: float) -> None:
    full = f"{_M}.{key}"
    lst = state.get(full, [])
    lst.append(value)
    state[full] = lst


# ---------------------------------------------------------------------------
# Model callbacks
# ---------------------------------------------------------------------------

def before_model_callback(callback_context, llm_request):
    """Log and time model invocation start."""
    agent_name = getattr(callback_context, "agent_name", "unknown")
    timer_key = f"{_TIMER_PREFIX}model.{agent_name}"
    callback_context.state[timer_key] = time.perf_counter()

    model = getattr(llm_request, "model", None) or "unknown"
    identity = _context_identity(callback_context)
    logger.info(
        "[MONITOR] Model call started | user=%s role=%s session=%s agent=%s model=%s",
        identity["user_id"], identity["role"], identity["session_id"], agent_name, model,
    )
    return None


def after_model_callback(callback_context, llm_response):
    """Log model latency, token usage, and accumulate metrics."""
    agent_name = getattr(callback_context, "agent_name", "unknown")
    state = callback_context.state

    timer_key = f"{_TIMER_PREFIX}model.{agent_name}"
    start = state.get(timer_key, None)
    latency_ms = (time.perf_counter() - start) * 1000 if start else 0.0

    _inc(state, "model_calls")
    _append(state, "model_latencies", latency_ms)

    usage = getattr(llm_response, "usage_metadata", None)
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cached_tokens = 0

    if usage:
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

        _inc(state, "prompt_tokens", prompt_tokens)
        _inc(state, "completion_tokens", completion_tokens)
        _inc(state, "total_tokens", total_tokens)
        _inc(state, "cached_tokens", cached_tokens)

    identity = _context_identity(callback_context)
    logger.info(
        "[MONITOR] Model call completed | user=%s role=%s session=%s agent=%s latency=%.1fms "
        "tokens(prompt=%d, completion=%d, total=%d, cached=%d)",
        identity["user_id"], identity["role"], identity["session_id"],
        agent_name, latency_ms, prompt_tokens, completion_tokens,
        total_tokens, cached_tokens,
    )

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "model_call",
        **identity,
        "agent_name": agent_name,
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }
    _log_event(event)
    log_audit_event(
        action="adk_model_call",
        user_id=identity["user_id"],
        role=identity["role"],
        metadata={
            "session_id": identity["session_id"],
            "agent_name": agent_name,
            "latency_ms": round(latency_ms, 1),
            "total_tokens": total_tokens,
        },
    )
    return None


def on_model_error_callback(callback_context, llm_request, error):
    """Log model errors and increment error counter."""
    agent_name = getattr(callback_context, "agent_name", "unknown")
    state = callback_context.state

    timer_key = f"{_TIMER_PREFIX}model.{agent_name}"
    state.get(timer_key, None)

    _inc(state, "model_errors")

    error_type = type(error).__name__
    identity = _context_identity(callback_context)
    logger.error(
        "[MONITOR] Model error | user=%s role=%s session=%s agent=%s error_type=%s error=%s",
        identity["user_id"], identity["role"], identity["session_id"],
        agent_name, error_type, str(error)[:200],
    )

    _log_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "model_error",
        **identity,
        "agent_name": agent_name,
        "error_type": error_type,
        "error_message": str(error)[:200],
    })
    log_audit_event(
        action="adk_model_error",
        user_id=identity["user_id"],
        role=identity["role"],
        status="failed",
        metadata={
            "session_id": identity["session_id"],
            "agent_name": agent_name,
            "error_type": error_type,
        },
    )
    return None


# ---------------------------------------------------------------------------
# Tool callbacks
# ---------------------------------------------------------------------------

def before_tool_callback(tool, args, tool_context):
    """Log and time tool execution start."""
    tool_name = getattr(tool, "name", str(tool))
    timer_key = f"{_TIMER_PREFIX}tool.{tool_name}"
    tool_context.state[timer_key] = time.perf_counter()

    args_summary = {k: str(v)[:100] for k, v in (args or {}).items()}
    identity = _context_identity(tool_context)
    logger.info(
        "[MONITOR] Tool call started | user=%s role=%s session=%s tool=%s args=%s",
        identity["user_id"], identity["role"], identity["session_id"], tool_name, args_summary,
    )
    return None


def after_tool_callback(tool, args, tool_context, tool_response):
    """Log tool latency, response status, and accumulate metrics."""
    tool_name = getattr(tool, "name", str(tool))
    state = tool_context.state

    timer_key = f"{_TIMER_PREFIX}tool.{tool_name}"
    start = state.get(timer_key, None)
    latency_ms = (time.perf_counter() - start) * 1000 if start else 0.0

    _inc(state, "tool_calls")
    _append(state, "tool_latencies", latency_ms)

    status = "unknown"
    if isinstance(tool_response, dict):
        status = tool_response.get("status", "unknown")

    identity = _context_identity(tool_context)
    logger.info(
        "[MONITOR] Tool call completed | user=%s role=%s session=%s tool=%s latency=%.1fms status=%s",
        identity["user_id"], identity["role"], identity["session_id"],
        tool_name, latency_ms, status,
    )

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "tool_call",
        **identity,
        "agent_name": getattr(tool_context, "agent_name", "unknown"),
        "tool_name": tool_name,
        "latency_ms": round(latency_ms, 1),
        "status": status,
    }
    _log_event(event)
    log_audit_event(
        action="adk_tool_call",
        user_id=identity["user_id"],
        role=identity["role"],
        metadata={
            "session_id": identity["session_id"],
            "tool_name": tool_name,
            "latency_ms": round(latency_ms, 1),
            "status": status,
        },
    )
    return None


def on_tool_error_callback(tool, args, tool_context, error):
    """Log tool errors and increment error counter."""
    tool_name = getattr(tool, "name", str(tool))
    state = tool_context.state

    timer_key = f"{_TIMER_PREFIX}tool.{tool_name}"
    state.get(timer_key, None)

    _inc(state, "tool_errors")

    error_type = type(error).__name__
    identity = _context_identity(tool_context)
    logger.error(
        "[MONITOR] Tool error | user=%s role=%s session=%s tool=%s error_type=%s error=%s",
        identity["user_id"], identity["role"], identity["session_id"],
        tool_name, error_type, str(error)[:200],
    )

    _log_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "tool_error",
        **identity,
        "agent_name": getattr(tool_context, "agent_name", "unknown"),
        "tool_name": tool_name,
        "error_type": error_type,
        "error_message": str(error)[:200],
    })
    log_audit_event(
        action="adk_tool_error",
        user_id=identity["user_id"],
        role=identity["role"],
        status="failed",
        metadata={
            "session_id": identity["session_id"],
            "tool_name": tool_name,
            "error_type": error_type,
        },
    )
    return None


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

def get_session_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract accumulated metrics from session state."""
    metrics = {}
    prefix = f"{_M}."
    for key, value in state.items():
        if key.startswith(prefix):
            metrics[key[len(prefix):]] = value

    if "model_latencies" in metrics:
        latencies = sorted(metrics["model_latencies"])
        n = len(latencies)
        if n > 0:
            metrics["model_latency_p50"] = latencies[n // 2]
            metrics["model_latency_p95"] = latencies[int(n * 0.95)]
            metrics["model_latency_avg"] = sum(latencies) / n

    if "tool_latencies" in metrics:
        latencies = sorted(metrics["tool_latencies"])
        n = len(latencies)
        if n > 0:
            metrics["tool_latency_p50"] = latencies[n // 2]
            metrics["tool_latency_p95"] = latencies[int(n * 0.95)]
            metrics["tool_latency_avg"] = sum(latencies) / n

    return metrics


def print_metrics_summary(state: Dict[str, Any]) -> str:
    """Format accumulated metrics as a human-readable summary."""
    m = get_session_metrics(state)
    if not m:
        return "No metrics collected."

    lines = [
        "=" * 60,
        "  SESSION METRICS SUMMARY",
        "=" * 60,
        "",
        "  Model Calls",
        f"    Total calls:      {m.get('model_calls', 0)}",
        f"    Errors:           {m.get('model_errors', 0)}",
        f"    Avg latency:      {m.get('model_latency_avg', 0):.1f} ms",
        f"    P50 latency:      {m.get('model_latency_p50', 0):.1f} ms",
        f"    P95 latency:      {m.get('model_latency_p95', 0):.1f} ms",
        "",
        "  Token Usage",
        f"    Prompt tokens:    {m.get('prompt_tokens', 0)}",
        f"    Completion tokens:{m.get('completion_tokens', 0)}",
        f"    Total tokens:     {m.get('total_tokens', 0)}",
        f"    Cached tokens:    {m.get('cached_tokens', 0)}",
        "",
        "  Tool Calls",
        f"    Total calls:      {m.get('tool_calls', 0)}",
        f"    Errors:           {m.get('tool_errors', 0)}",
        f"    Avg latency:      {m.get('tool_latency_avg', 0):.1f} ms",
        f"    P50 latency:      {m.get('tool_latency_p50', 0):.1f} ms",
        f"    P95 latency:      {m.get('tool_latency_p95', 0):.1f} ms",
        "",
        "=" * 60,
    ]

    summary = "\n".join(lines)
    logger.info("\n%s", summary)
    return summary
