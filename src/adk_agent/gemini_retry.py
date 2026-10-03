"""Retry helpers for transient Gemini quota errors."""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

_rate_limit_lock = threading.Lock()
_next_call_at = 0.0


def _wait_for_rate_limit(min_interval_seconds: float) -> None:
    """Pace Gemini calls in-process to avoid low RPM quota bursts."""
    if min_interval_seconds <= 0:
        return

    global _next_call_at
    with _rate_limit_lock:
        now = time.monotonic()
        wait_seconds = max(0.0, _next_call_at - now)
        if wait_seconds > 0:
            logger.info("Pacing Gemini call. Waiting %.2f seconds...", wait_seconds)
            time.sleep(wait_seconds)
            now = time.monotonic()
        _next_call_at = max(now, _next_call_at) + min_interval_seconds


def _error_code(exc: BaseException) -> Any:
    return getattr(exc, "code", None) or getattr(exc, "status_code", None)


def _is_quota_error(exc: BaseException) -> bool:
    code = _error_code(exc)
    return code == 429 or code == "429"


def call_gemini_with_retry(
    client: Any,
    model: str,
    prompt: Any,
    retries: int = 5,
    backoff_factor: int = 2,
    min_interval_seconds: float = 4.0,
) -> Any:
    """Call Gemini generate_content with exponential backoff on 429 errors."""
    delay = 1
    for attempt in range(retries):
        try:
            _wait_for_rate_limit(min_interval_seconds)
            return client.models.generate_content(
                model=model,
                contents=prompt,
            )
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            if attempt == retries - 1:
                break
            logger.warning("Quota exceeded. Retrying in %s seconds...", delay)
            time.sleep(delay)
            delay *= backoff_factor
    raise RuntimeError("Max retries exceeded")


def _retrying_stream(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    retries: int,
    backoff_factor: int,
    min_interval_seconds: float,
) -> Any:
    delay = 1
    for attempt in range(retries):
        try:
            _wait_for_rate_limit(min_interval_seconds)
            yield from func(*args, **kwargs)
            return
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            if attempt == retries - 1:
                break
            logger.warning("Quota exceeded. Retrying in %s seconds...", delay)
            time.sleep(delay)
            delay *= backoff_factor
    raise RuntimeError("Max retries exceeded")


def _with_retry(
    func: Callable[..., Any],
    retries: int,
    backoff_factor: int,
    min_interval_seconds: float,
) -> Callable[..., Any]:
    if inspect.isasyncgenfunction(func):
        async def async_stream_wrapper(*args: Any, **kwargs: Any) -> Any:
            import asyncio

            delay = 1
            for attempt in range(retries):
                try:
                    await asyncio.to_thread(_wait_for_rate_limit, min_interval_seconds)
                    async for item in func(*args, **kwargs):
                        yield item
                    return
                except Exception as exc:
                    if not _is_quota_error(exc):
                        raise
                    if attempt == retries - 1:
                        break
                    logger.warning("Quota exceeded. Retrying in %s seconds...", delay)
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
            raise RuntimeError("Max retries exceeded")

        setattr(async_stream_wrapper, "_gemini_retry_wrapped", True)
        return async_stream_wrapper

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            import asyncio

            delay = 1
            for attempt in range(retries):
                try:
                    await asyncio.to_thread(_wait_for_rate_limit, min_interval_seconds)
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if not _is_quota_error(exc):
                        raise
                    if attempt == retries - 1:
                        break
                    logger.warning("Quota exceeded. Retrying in %s seconds...", delay)
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
            raise RuntimeError("Max retries exceeded")

        setattr(async_wrapper, "_gemini_retry_wrapped", True)
        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if inspect.isgeneratorfunction(func):
            return _retrying_stream(
                func,
                args,
                kwargs,
                retries,
                backoff_factor,
                min_interval_seconds,
            )

        delay = 1
        for attempt in range(retries):
            try:
                _wait_for_rate_limit(min_interval_seconds)
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                if not _is_quota_error(exc):
                    raise
                if attempt == retries - 1:
                    break
                logger.warning("Quota exceeded. Retrying in %s seconds...", delay)
                time.sleep(delay)
                delay *= backoff_factor
        raise RuntimeError("Max retries exceeded")

    setattr(wrapper, "_gemini_retry_wrapped", True)
    return wrapper


def install_gemini_retry_patch(
    retries: int = 5,
    backoff_factor: int = 2,
    min_interval_seconds: float = 4.0,
) -> None:
    """Patch google-genai calls used by ADK with pacing and 429 retries."""
    try:
        from google.genai import models as genai_models
    except Exception:
        logger.debug("google-genai is not available; Gemini retry patch skipped")
        return

    for class_name in ("Models", "AsyncModels"):
        model_class = getattr(genai_models, class_name, None)
        if model_class is None:
            continue

        for method_name in ("generate_content", "generate_content_stream"):
            generate_content = getattr(model_class, method_name, None)
            if generate_content is None or getattr(generate_content, "_gemini_retry_wrapped", False):
                continue

            setattr(
                model_class,
                method_name,
                _with_retry(
                    generate_content,
                    retries=retries,
                    backoff_factor=backoff_factor,
                    min_interval_seconds=min_interval_seconds,
                ),
            )
            logger.info(
                "Installed Gemini retry/rate-limit patch for %s.%s",
                class_name,
                method_name,
            )
