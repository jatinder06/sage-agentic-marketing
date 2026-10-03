"""SQLite persistence for ADK monitoring events and demo run summaries."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "adk_monitoring.db"

_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DEFAULT_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()


def init_monitoring_store() -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                role TEXT,
                session_id TEXT,
                agent_name TEXT,
                tool_name TEXT,
                latency_ms REAL,
                status TEXT,
                total_tokens INTEGER,
                error_type TEXT,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_monitoring_events_timestamp
            ON monitoring_events(timestamp)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS demo_runs (
                story TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                elapsed_seconds REAL NOT NULL,
                event_count INTEGER NOT NULL,
                completed_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_cost_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                story TEXT,
                workflow_step TEXT,
                endpoint TEXT NOT NULL,
                model_name TEXT NOT NULL,
                provider_type TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                latency_ms REAL,
                pricing_basis_model TEXT NOT NULL,
                gemini_input_usd_per_1m REAL NOT NULL,
                gemini_output_usd_per_1m REAL NOT NULL,
                gemini_equivalent_cost_usd REAL NOT NULL,
                actual_api_cost_usd REAL NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_monitoring_event(event: Dict[str, Any]) -> None:
    init_monitoring_store()
    payload = dict(event)
    timestamp = str(payload.get("timestamp") or _utc_now())
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO monitoring_events (
                timestamp, event_type, user_id, role, session_id, agent_name,
                tool_name, latency_ms, status, total_tokens, error_type, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                payload.get("event_type", "unknown"),
                payload.get("user_id"),
                payload.get("role"),
                payload.get("session_id"),
                payload.get("agent_name"),
                payload.get("tool_name"),
                payload.get("latency_ms"),
                payload.get("status"),
                payload.get("total_tokens"),
                payload.get("error_type"),
                json.dumps(payload, default=str),
            ),
        )
        conn.commit()


def load_monitoring_events(limit: int = 5000) -> List[Dict[str, Any]]:
    init_monitoring_store()
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            """
            SELECT payload_json
            FROM monitoring_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    events = [json.loads(row["payload_json"]) for row in rows]
    events.reverse()
    return events


def clear_monitoring_events() -> None:
    init_monitoring_store()
    with _LOCK, _connect() as conn:
        conn.execute("DELETE FROM monitoring_events")
        conn.commit()


def save_demo_run(
    story: str,
    elapsed_seconds: float,
    event_count: int,
    outputs: Dict[str, Any] | None = None,
) -> None:
    init_monitoring_store()
    record = {
        "story": story,
        "status": "completed",
        "elapsed_seconds": elapsed_seconds,
        "event_count": event_count,
        "completed_at": _utc_now(),
        "outputs": outputs or {},
    }
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO demo_runs (
                story, status, elapsed_seconds, event_count, completed_at, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(story) DO UPDATE SET
                status=excluded.status,
                elapsed_seconds=excluded.elapsed_seconds,
                event_count=excluded.event_count,
                completed_at=excluded.completed_at,
                payload_json=excluded.payload_json
            """,
            (
                story,
                record["status"],
                elapsed_seconds,
                event_count,
                record["completed_at"],
                json.dumps(record, default=str),
            ),
        )
        conn.commit()


def get_demo_run(story: str) -> Dict[str, Any] | None:
    init_monitoring_store()
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT payload_json FROM demo_runs WHERE story = ?",
            (story,),
        ).fetchone()
    return json.loads(row["payload_json"]) if row else None


def clear_demo_run(story: str) -> None:
    init_monitoring_store()
    with _LOCK, _connect() as conn:
        conn.execute("DELETE FROM demo_runs WHERE story = ?", (story,))
        conn.commit()


def clear_demo_runs() -> None:
    init_monitoring_store()
    with _LOCK, _connect() as conn:
        conn.execute("DELETE FROM demo_runs")
        conn.commit()


def save_model_cost_event(event: Dict[str, Any]) -> None:
    init_monitoring_store()
    payload = dict(event)
    timestamp = str(payload.get("timestamp") or _utc_now())
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO model_cost_events (
                timestamp, story, workflow_step, endpoint, model_name,
                provider_type, input_tokens, output_tokens, total_tokens,
                latency_ms, pricing_basis_model, gemini_input_usd_per_1m,
                gemini_output_usd_per_1m, gemini_equivalent_cost_usd,
                actual_api_cost_usd, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                payload.get("story"),
                payload.get("workflow_step"),
                payload.get("endpoint"),
                payload.get("model_name"),
                payload.get("provider_type"),
                int(payload.get("input_tokens") or 0),
                int(payload.get("output_tokens") or 0),
                int(payload.get("total_tokens") or 0),
                payload.get("latency_ms"),
                payload.get("pricing_basis_model"),
                float(payload.get("gemini_input_usd_per_1m") or 0.0),
                float(payload.get("gemini_output_usd_per_1m") or 0.0),
                float(payload.get("gemini_equivalent_cost_usd") or 0.0),
                float(payload.get("actual_api_cost_usd") or 0.0),
                json.dumps(payload, default=str),
            ),
        )
        conn.commit()


def load_model_cost_events(limit: int = 5000, story: str | None = None) -> List[Dict[str, Any]]:
    init_monitoring_store()
    with _LOCK, _connect() as conn:
        if story:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM model_cost_events
                WHERE story = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (story, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM model_cost_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    events = [json.loads(row["payload_json"]) for row in rows]
    events.reverse()
    return events


def clear_model_cost_events(story: str | None = None) -> None:
    init_monitoring_store()
    with _LOCK, _connect() as conn:
        if story:
            conn.execute("DELETE FROM model_cost_events WHERE story = ?", (story,))
        else:
            conn.execute("DELETE FROM model_cost_events")
        conn.commit()
