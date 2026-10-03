"""AI Marketing Multi-Agent System — Google ADK entry point.

Usage:
    adk web src.adk_agent
    adk run src.adk_agent
"""

__all__ = ["root_agent"]


def __getattr__(name: str):
    if name == "root_agent":
        from src.adk_agent.agent import root_agent

        return root_agent
    raise AttributeError(name)
