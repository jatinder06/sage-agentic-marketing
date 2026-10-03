"""Logging setup for ADK agent surfaces."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_adk_logging() -> None:
    """Configure console and optional file logging once."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_adk_logging_configured", False):
        return

    level_name = os.getenv("ADK_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.getenv(
        "ADK_LOG_FORMAT",
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    root_logger.setLevel(level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)

    if os.getenv("ADK_LOG_TO_FILE", "true").lower() == "true":
        project_root = Path(__file__).resolve().parents[2]
        log_file = Path(os.getenv("ADK_LOG_FILE", str(project_root / "logs" / "adk_agent.log")))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    setattr(root_logger, "_adk_logging_configured", True)
