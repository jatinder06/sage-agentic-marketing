"""Lightweight user authentication for the ADK dashboard."""

from __future__ import annotations

import hmac
import logging
import os
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

ROLE_ADMIN = "admin"
ROLE_MARKETING_MANAGER = "marketing_manager"
ROLE_DATA_SCIENTIST = "data_scientist"
ROLE_VIEWER = "viewer"

_VALID_ROLES = {
    ROLE_ADMIN,
    ROLE_MARKETING_MANAGER,
    ROLE_DATA_SCIENTIST,
    ROLE_VIEWER,
}


@dataclass(frozen=True)
class UserContext:
    user_id: str
    role: str


def _parse_users(raw: str) -> Dict[str, tuple[str, str]]:
    users: Dict[str, tuple[str, str]] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            logger.warning("Ignoring malformed ADK_DASHBOARD_USERS entry")
            continue
        username, password, role = (part.strip() for part in parts)
        role = role.lower()
        if not username or not password or role not in _VALID_ROLES:
            logger.warning("Ignoring invalid ADK_DASHBOARD_USERS entry for user=%s", username)
            continue
        users[username] = (password, role)
    return users


def configured_users() -> Dict[str, tuple[str, str]]:
    """Return username -> (password, role) mapping.

    Configure with:
        ADK_DASHBOARD_USERS="admin:admin123:data_scientist,marketing:mkt2024:marketing_manager"
    """
    raw = os.getenv("ADK_DASHBOARD_USERS", "").strip()
    if raw:
        return _parse_users(raw)

    logger.warning(
        "ADK_DASHBOARD_USERS not configured; using local demo user credentials."
    )
    return {
        "admin": ("admin123", ROLE_DATA_SCIENTIST),
        "marketing": ("mkt2024", ROLE_MARKETING_MANAGER),
        "viewer": ("view123", ROLE_VIEWER),
    }


def authenticate_user(username: str, password: str) -> UserContext | None:
    users = configured_users()
    stored = users.get(username)
    if not stored:
        return None

    expected_password, role = stored
    if not hmac.compare_digest(password, expected_password):
        return None
    return UserContext(user_id=username, role=role)
