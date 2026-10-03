"""User-scoped audit and trace events for ADK surfaces."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_audit_log: List[Dict[str, Any]] = []
_AUDIT_LOG_MAX = 5000
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_FILE = _PROJECT_ROOT / "logs" / "adk_audit.log"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_file(record: Dict[str, Any]) -> None:
    try:
        _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.debug("Could not write ADK audit log: %s", exc)


def _load_file() -> List[Dict[str, Any]]:
    if not _AUDIT_FILE.exists():
        return []

    records: List[Dict[str, Any]] = []
    try:
        with _AUDIT_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed ADK audit log line")
    except Exception as exc:
        logger.warning("Could not read ADK audit log: %s", exc)
    return records[-_AUDIT_LOG_MAX:]


def log_audit_event(
    *,
    action: str,
    user_id: str = "anonymous",
    role: str = "unknown",
    status: str = "success",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    record = {
        "timestamp": _utc_now(),
        "action": action,
        "user_id": user_id,
        "role": role,
        "status": status,
        "metadata": metadata or {},
    }
    _audit_log.append(record)
    if len(_audit_log) > _AUDIT_LOG_MAX:
        del _audit_log[: len(_audit_log) - _AUDIT_LOG_MAX]
    _append_file(record)
    logger.info(
        "[AUDIT] action=%s user=%s role=%s status=%s",
        action,
        user_id,
        role,
        status,
    )
    return record


def get_audit_log() -> List[Dict[str, Any]]:
    records = _load_file()
    if records:
        return records
    return list(_audit_log)


def clear_audit_log() -> None:
    _audit_log.clear()
    try:
        _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _AUDIT_FILE.write_text("", encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not clear ADK audit log file: %s", exc)
