"""OpenTelemetry configuration for the ADK multi-agent system.

Configures tracing, metrics, and logging exporters using ADK's built-in
OTel provider setup.  Supports console export for development and OTLP
HTTP export for production observability backends (Jaeger, Grafana Tempo).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def setup_telemetry(
    enable_console: bool = True,
    otlp_endpoint: Optional[str] = None,
    capture_content: bool = False,
) -> None:
    """Initialise OpenTelemetry providers for the ADK agent system.

    Args:
        enable_console: Write spans to stdout (development mode).
        otlp_endpoint: OTLP HTTP endpoint URL for production export.
            Overrides OTEL_EXPORTER_OTLP_ENDPOINT env var if set.
        capture_content: Capture full message content in spans (verbose).
    """
    from google.adk.telemetry.setup import maybe_set_otel_providers, OTelHooks
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
        BatchSpanProcessor,
    )

    if capture_content:
        os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

    if otlp_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint

    span_processors = []

    if enable_console:
        span_processors.append(SimpleSpanProcessor(ConsoleSpanExporter()))
        logger.info("[TELEMETRY] Console span exporter enabled")

    if otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            endpoint = otlp_endpoint or os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
            span_processors.append(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            logger.info("[TELEMETRY] OTLP span exporter enabled → %s", endpoint)
        except ImportError:
            logger.warning(
                "[TELEMETRY] opentelemetry-exporter-otlp-proto-http not installed; "
                "skipping OTLP export"
            )

    if span_processors:
        hooks = OTelHooks(span_processors=span_processors)
        maybe_set_otel_providers(otel_hooks_to_setup=[hooks])
        logger.info("[TELEMETRY] OpenTelemetry providers initialised")
    else:
        logger.info("[TELEMETRY] No exporters configured; telemetry disabled")
