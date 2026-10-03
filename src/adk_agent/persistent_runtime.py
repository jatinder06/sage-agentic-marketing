"""Persistent ADK runner/session helpers.

ADK's InMemoryRunner is convenient for demos, but its sessions disappear when
the process restarts. This module prefers ADK's DatabaseSessionService backed by
SQLite, while keeping an InMemoryRunner fallback for environments where the
installed ADK version does not expose the database service.
"""

from __future__ import annotations

import logging
import os
import inspect
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_DB = PROJECT_ROOT / "data" / "adk_sessions.db"
_LAST_RUNTIME_STATUS: Dict[str, Any] = {
    "persistent": False,
    "service": "not_initialized",
    "db_url": None,
    "error": None,
}


def _session_db_url() -> str:
    raw = os.getenv("ADK_SESSION_DB_URL", "").strip()
    if raw:
        return raw
    DEFAULT_SESSION_DB.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_SESSION_DB}"


def create_persistent_runner(agent: Any, app_name: str) -> Any:
    """Create an ADK Runner with persistent SQLite sessions when possible."""
    db_url = _session_db_url()
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import DatabaseSessionService

        session_service = DatabaseSessionService(db_url=db_url)
        logger.info("Using ADK DatabaseSessionService at %s", db_url)
        _LAST_RUNTIME_STATUS.update(
            {
                "persistent": True,
                "service": "DatabaseSessionService",
                "db_url": db_url,
                "error": None,
            }
        )
        return Runner(
            agent=agent,
            app_name=app_name,
            session_service=session_service,
        )
    except Exception as exc:
        logger.warning(
            "Falling back to InMemoryRunner; persistent ADK sessions unavailable: %s",
            exc,
        )
        _LAST_RUNTIME_STATUS.update(
            {
                "persistent": False,
                "service": "InMemoryRunner",
                "db_url": db_url,
                "error": str(exc),
            }
        )
        from google.adk.runners import InMemoryRunner

        return InMemoryRunner(agent=agent, app_name=app_name)


def get_runtime_status() -> Dict[str, Any]:
    """Return the most recent runner/session persistence status."""
    return dict(_LAST_RUNTIME_STATUS)


async def get_or_create_session(
    runner: Any,
    *,
    user_id: str,
    session_id: str,
    role: str = "unknown",
    extra_state: Dict[str, Any] | None = None,
) -> Any:
    """Get or create an ADK session with identity state."""
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )
    session_state = {
        "adk.user_id": user_id,
        "adk.role": role,
        "adk.session_id": session_id,
        **(extra_state or {}),
    }

    if session:
        if hasattr(session, "state") and isinstance(session.state, dict):
            changed = False
            for key, value in session_state.items():
                if key not in session.state:
                    session.state[key] = value
                    changed = True
            update_session = getattr(runner.session_service, "update_session", None)
            if changed and update_session:
                try:
                    result = update_session(session=session)
                except TypeError:
                    result = update_session(session)
                if inspect.isawaitable(result):
                    await result
        return session

    try:
        return await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
            state=session_state,
        )
    except TypeError:
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        session = await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session and hasattr(session, "state"):
            for key, value in session_state.items():
                session.state[key] = value
        return session
