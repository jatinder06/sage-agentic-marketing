"""Standalone runner for the ADK multi-agent system with full observability.

Usage:
    python -m src.adk_agent.run "Analyze the sentiment of: Great product!"
    python -m src.adk_agent.run --telemetry "Is this customer at risk? tenure=3, contract=Month-to-month"
    python -m src.adk_agent.run --otlp http://localhost:4318 "Segment this customer..."
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import List

from google.genai import types

from src.adk_agent.audit import log_audit_event
from src.adk_agent.agent import root_agent
from src.adk_agent.callbacks import print_metrics_summary
from src.adk_agent.logging_config import configure_adk_logging
from src.adk_agent.persistent_runtime import create_persistent_runner, get_or_create_session

logger = logging.getLogger(__name__)


async def run_query(
    runner,
    user_message: str,
    user_id: str = "monitoring_user",
    session_id: str = "monitoring_session",
    role: str = "cli_user",
) -> List:
    """Run a single user query through the agent system and return events."""
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    await get_or_create_session(
        runner,
        user_id=user_id,
        session_id=session_id,
        role=role,
    )

    events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        events.append(event)

        author = getattr(event, "author", "")
        text_parts = []
        if hasattr(event, "content") and event.content:
            for part in getattr(event.content, "parts", []):
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

        if text_parts:
            print(f"\n[{author}] {' '.join(text_parts)}")

    return events


def _extract_session_state(runner, user_id: str, session_id: str) -> dict:
    """Extract session state from the runner's session service."""
    try:
        session_service = runner.session_service
        loop = asyncio.get_event_loop()
        session = loop.run_until_complete(
            session_service.get_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id,
            )
        )
        if session:
            return session.state.to_dict()
    except Exception as e:
        logger.debug("Could not extract session state: %s", e)
    return {}


async def run_with_monitoring(
    user_message: str,
    enable_telemetry: bool = False,
    otlp_endpoint: str | None = None,
    capture_content: bool = False,
    user_id: str = "monitoring_user",
    session_id: str = "monitoring_session",
    role: str = "cli_user",
) -> None:
    """Run a query with full monitoring and print a metrics summary."""
    if enable_telemetry or otlp_endpoint:
        from src.adk_agent.telemetry_setup import setup_telemetry
        setup_telemetry(
            enable_console=enable_telemetry,
            otlp_endpoint=otlp_endpoint,
            capture_content=capture_content,
        )

    runner = create_persistent_runner(agent=root_agent, app_name="ai_marketing")

    print(f"\nQuery: {user_message}")
    print("-" * 60)

    log_audit_event(
        action="cli_query_start",
        user_id=user_id,
        role=role,
        metadata={"session_id": session_id, "query_preview": user_message[:200]},
    )
    events = await run_query(runner, user_message, user_id, session_id, role)
    log_audit_event(
        action="cli_query_complete",
        user_id=user_id,
        role=role,
        metadata={"session_id": session_id, "events": len(events)},
    )

    print("\n")

    try:
        session = await runner.session_service.get_session(
            app_name="ai_marketing",
            user_id=user_id,
            session_id=session_id,
        )
        if session:
            summary = print_metrics_summary(session.state.to_dict())
            print(summary)
    except Exception as e:
        logger.warning("Could not retrieve session metrics: %s", e)

    print(f"\nTotal events: {len(events)}")


def main():
    configure_adk_logging()
    parser = argparse.ArgumentParser(
        description="Run the AI Marketing agent system with monitoring"
    )
    parser.add_argument("message", help="User message to send to the agent")
    parser.add_argument(
        "--telemetry", action="store_true",
        help="Enable OpenTelemetry console span export",
    )
    parser.add_argument(
        "--otlp", type=str, default=None,
        help="OTLP HTTP endpoint for span export (e.g. http://localhost:4318)",
    )
    parser.add_argument(
        "--capture-content", action="store_true",
        help="Capture full message content in telemetry spans",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument("--user-id", default="monitoring_user", help="User ID for audit/tracing")
    parser.add_argument("--role", default="cli_user", help="User role for audit/tracing")
    parser.add_argument("--session-id", default="monitoring_session", help="ADK session ID")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(run_with_monitoring(
        user_message=args.message,
        enable_telemetry=args.telemetry,
        otlp_endpoint=args.otlp,
        capture_content=args.capture_content,
        user_id=args.user_id,
        session_id=args.session_id,
        role=args.role,
    ))


if __name__ == "__main__":
    main()
