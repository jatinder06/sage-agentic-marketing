"""Digital Marketing Intelligence System dashboard for the ADK multi-agent system.

Provides real-time observability into model calls, tool executions, token usage,
latency distributions, error rates, and live agent testing.

Run (two terminals):

    # Terminal 1 — model inference server
    python -m src.adk_agent.model_server

    # Terminal 2 — this dashboard
    streamlit run src/adk_agent/dashboard.py
"""

from __future__ import annotations

# Prevent segfault: loky semaphore leak (see docs/DEVELOPMENT_ISSUES_AND_FIXES.md #2)
import multiprocessing.resource_tracker as _rt
_original_register = _rt.register
def _no_sem_register(name, rtype):
    if rtype == "semaphore":
        return
    _original_register(name, rtype)
_rt.register = _no_sem_register
del _rt, _original_register

import asyncio
import html
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Force standard asyncio loop (uvloop can't be patched by nest_asyncio)
import nest_asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)
nest_asyncio.apply(_loop)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adk_agent.callbacks import (
    clear_event_log,
    get_event_log,
    get_session_metrics,
)
from src.adk_agent.cost_tracking import summarize_cost_evidence
from src.adk_agent.audit import clear_audit_log, get_audit_log, log_audit_event
from src.adk_agent.auth import (
    ROLE_ADMIN,
    ROLE_DATA_SCIENTIST,
    ROLE_MARKETING_MANAGER,
    ROLE_VIEWER,
    authenticate_user,
)
from src.adk_agent.logging_config import configure_adk_logging
from src.adk_agent.monitoring_store import load_model_cost_events
from src.adk_agent.tools.model_client import server_healthy

configure_adk_logging()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Digital Marketing Intelligence System",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(20, 184, 166, 0.10), transparent 26%),
                radial-gradient(circle at 78% 6%, rgba(59, 130, 246, 0.12), transparent 24%),
                linear-gradient(135deg, #0B1020 0%, #111827 48%, #0F172A 100%);
        }

        .block-container {
            max-width: 1480px;
            padding-top: 4.2rem;
            padding-bottom: 2.5rem;
        }

        header[data-testid="stHeader"] {
            background: #0B1020;
            border-bottom: 1px solid rgba(148, 163, 184, 0.20);
        }

        div[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.96));
            border-right: 1px solid rgba(148, 163, 184, 0.24);
        }

        .sidebar-brand {
            font-size: 1.28rem;
            font-weight: 850;
            line-height: 1.15;
            color: #F8FAFC;
            margin: 0.2rem 0 1.05rem 0;
        }

        .overview-card {
            min-height: 122px;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-top: 4px solid var(--card-accent);
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.075), rgba(255, 255, 255, 0.025));
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.22);
            padding: 0.9rem 0.95rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .overview-card-label {
            color: #CBD5E1;
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.45rem;
        }

        .overview-card-value {
            color: #F8FAFC;
            font-size: 1.78rem;
            line-height: 1.05;
            font-weight: 850;
            overflow-wrap: anywhere;
        }

        .overview-card-hint {
            color: #94A3B8;
            font-size: 0.72rem;
            line-height: 1.25;
            margin-top: 0.7rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.18);
        }

        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 10px;
            min-height: 2.55rem;
            font-weight: 800;
            border: 1px solid rgba(148, 163, 184, 0.35);
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.30);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if not server_healthy():
    st.warning(
        "**Model server not running.** Start it in a separate terminal first:\n\n"
        "```bash\npython -m src.adk_agent.model_server\n```",
        icon="⚠️",
    )


def _current_user() -> Dict[str, str] | None:
    return st.session_state.get("adk_user")


def _require_login() -> Dict[str, str]:
    user = _current_user()
    if user:
        return user

    st.markdown(
        """
        <style>
            /* Login page layout aligned with Figure 4.8 wireframe */
            div[data-testid="stSidebar"] {
                display: none;
            }

            .block-container {
                max-width: 1180px;
                padding-top: 2.2rem;
            }

            .login-title {
                text-align: center;
                font-size: 2.65rem;
                font-weight: 800;
                margin-bottom: 0.35rem;
                white-space: nowrap;
            }

            .login-subtitle {
                text-align: center;
                color: #9CA3AF;
                font-size: 1.05rem;
                margin-bottom: 1.15rem;
            }

            .login-brand {
                border: 2px solid #2E7D32;
                background: rgba(46, 125, 50, 0.16);
                border-radius: 12px;
                padding: 0.85rem 1rem;
                text-align: center;
                font-size: 1.35rem;
                font-weight: 800;
                color: #DFF7E1;
                margin-bottom: 1rem;
            }

            .login-demo {
                border: 2px solid #F9A825;
                background: rgba(249, 168, 37, 0.16);
                border-radius: 12px;
                padding: 0.85rem 1rem;
                text-align: center;
                color: #FFE9A6;
                font-weight: 700;
                margin-top: 0.9rem;
                line-height: 1.55;
            }

            div[data-testid="stForm"] {
                border: 2px solid rgba(120, 130, 150, 0.45);
                border-radius: 18px;
                padding: 1.6rem 2rem 1.35rem 2rem;
                background: rgba(17, 24, 39, 0.55);
                box-shadow: 0 18px 50px rgba(0, 0, 0, 0.22);
            }

            div[data-testid="stTextInput"] label {
                font-size: 1.02rem;
                font-weight: 700;
            }

            div[data-testid="stTextInput"] input {
                border-radius: 12px;
                min-height: 44px;
            }

            div[data-testid="stFormSubmitButton"] button {
                border-radius: 12px;
                min-height: 46px;
                font-size: 1.05rem;
                font-weight: 800;
                border: 2px solid #0D47A1;
                background: rgba(13, 71, 161, 0.28);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, login_col, _ = st.columns([0.65, 1.7, 0.65])
    with login_col:
        st.markdown('<div class="login-title">Digital Marketing Intelligence System</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Sign in to your account</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-brand">Marketing Intelligence Dashboard</div>', unsafe_allow_html=True)

        with st.form("adk_login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        st.markdown(
            '<div class="login-demo">Demo roles: Data Scientist, Marketing Manager, Viewer<br>'
            '<span style="font-weight:600;">admin/admin123 &nbsp;|&nbsp; marketing/mkt2024 &nbsp;|&nbsp; viewer/view123</span></div>',
            unsafe_allow_html=True,
        )

    if submitted:
        auth = authenticate_user(username.strip(), password)
        if auth:
            user = {"user_id": auth.user_id, "role": auth.role}
            st.session_state["adk_user"] = user
            log_audit_event(
                action="dashboard_login",
                user_id=auth.user_id,
                role=auth.role,
            )
            st.rerun()
        else:
            log_audit_event(
                action="dashboard_login",
                user_id=username.strip() or "unknown",
                role="unknown",
                status="failed",
            )
            st.error("Invalid user ID or password.")

    st.stop()


CURRENT_USER = _require_login()

CORE_PAGES = [
    "System Overview",
    "Model Performance",
    "Tool Performance",
    "Log Explorer",
    "User Audit",
    "Live Agent Tester",
]
DEMO_PAGES = [
    "Demo Simulation",
    "Demo Performance Dashboard",
    "Demo Before vs After Impact",
    "Demo Log Explorer",
    "Demo System Architecture",
]
ROLE_PAGES = {
    ROLE_ADMIN: CORE_PAGES + DEMO_PAGES,
    ROLE_DATA_SCIENTIST: CORE_PAGES + DEMO_PAGES,
    ROLE_MARKETING_MANAGER: [
        "System Overview",
        "Model Performance",
        "Tool Performance",
        "Log Explorer",
        "Live Agent Tester",
        *DEMO_PAGES,
    ],
    ROLE_VIEWER: [
        "System Overview",
        "Model Performance",
        "Tool Performance",
        "Log Explorer",
        "Demo Performance Dashboard",
        "Demo Before vs After Impact",
        "Demo Log Explorer",
        "Demo System Architecture",
    ],
}


def _current_role() -> str:
    return CURRENT_USER.get("role", ROLE_VIEWER)


def _can_administer() -> bool:
    return _current_role() in {ROLE_ADMIN, ROLE_DATA_SCIENTIST}


def _can_execute() -> bool:
    return _current_role() in {ROLE_ADMIN, ROLE_DATA_SCIENTIST, ROLE_MARKETING_MANAGER}


def _available_pages() -> List[str]:
    return ROLE_PAGES.get(_current_role(), ROLE_PAGES[ROLE_VIEWER])


def _supports_stretch_width(method_name: str) -> bool:
    """True when Streamlit supports width='stretch' for the method."""
    try:
        method = getattr(st, method_name)
        param = inspect.signature(method).parameters.get("width")
        return param is not None and isinstance(param.default, str)
    except Exception:
        return False


_DF_STRETCH_KW = (
    {"width": "stretch"}
    if _supports_stretch_width("dataframe")
    else {"use_container_width": True}
)
_PLOT_STRETCH_KW = (
    {"width": "stretch"}
    if _supports_stretch_width("plotly_chart")
    else {"use_container_width": True}
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_NAMES = [
    "ai_marketing_orchestrator",
    "sentiment_agent",
    "churn_agent",
    "segmentation_agent",
    "support_agent",
    "content_agent",
    "recommendation_agent",
]


def _events_df() -> pd.DataFrame:
    events = get_event_log()
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _model_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["event_type"].isin(["model_call", "model_error"])].copy()


def _tool_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["event_type"].isin(["tool_call", "tool_error"])].copy()


def _error_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["event_type"].str.contains("error", na=False)].copy()


def _safe_pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator > 0 else 0.0


def _overview_card(label: str, value: Any, hint: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="overview-card" style="--card-accent: {html.escape(accent)};">
            <div>
                <div class="overview-card-label">{html.escape(label)}</div>
                <div class="overview-card-value">{html.escape(str(value))}</div>
            </div>
            <div class="overview-card-hint">{html.escape(hint)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _story_monitoring_events(events: List[Dict[str, Any]], story: str) -> List[Dict[str, Any]]:
    prefix = f"{story}_"
    return [
        event
        for event in events
        if str(event.get("session_id", "")).startswith(prefix)
    ]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    '<div class="sidebar-brand">Digital Marketing<br>Intelligence System</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    _available_pages(),
    index=0,
)

DEMO_PAGE_SET = set(DEMO_PAGES)
is_demo_page = page in DEMO_PAGE_SET

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.selectbox(
        "Refresh interval", [5, 10, 30, 60], index=1
    )
    st.sidebar.caption(f"Refreshing every {refresh_interval}s")

st.sidebar.markdown("---")
st.sidebar.caption("Signed in as")
st.sidebar.write(f"**{CURRENT_USER['user_id']}**")
st.sidebar.caption(f"Role: `{CURRENT_USER['role']}`")
if st.sidebar.button("Sign out"):
    log_audit_event(
        action="dashboard_logout",
        user_id=CURRENT_USER["user_id"],
        role=CURRENT_USER["role"],
    )
    st.session_state.pop("adk_user", None)
    st.rerun()

st.sidebar.markdown("---")
event_count = len(get_event_log())
st.sidebar.metric("Events in buffer", event_count, help="Total events in ring buffer")
st.sidebar.caption("Backed by SQLite: data/adk_monitoring.db")
try:
    from src.adk_agent.persistent_runtime import get_runtime_status

    runtime_status = get_runtime_status()
    if runtime_status.get("persistent"):
        st.sidebar.success("ADK sessions: persistent SQLite")
    elif runtime_status.get("service") == "InMemoryRunner":
        st.sidebar.warning("ADK sessions: in-memory fallback")
        if runtime_status.get("error"):
            st.sidebar.caption(str(runtime_status["error"])[:160])
    else:
        st.sidebar.caption("ADK sessions: not initialized")
except Exception:
    st.sidebar.caption("ADK sessions: status unavailable")
st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

if _can_administer():
    if st.sidebar.button("Clear Event Log"):
        clear_event_log()
        log_audit_event(
            action="dashboard_clear_event_log",
            user_id=CURRENT_USER["user_id"],
            role=CURRENT_USER["role"],
        )
        st.sidebar.success("Event log cleared")
        st.rerun()

if is_demo_page:
    from src.adk_agent import demo as demo_pages

    demo_pages.inject_demo_theme()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Alpha & Beta Story Simulation**")
    demo_story = st.sidebar.radio(
        "Select Story",
        ["Alpha Story", "Beta Story", "Full Demo"],
        index=0,
    )
    st.session_state["demo_story"] = demo_story

    if st.sidebar.button("Export Demo HTML", key="main_export_demo_dashboard"):
        try:
            html_content = demo_pages.export_dashboard_html()
            st.sidebar.download_button(
                label="Download demo dashboard",
                data=html_content,
                file_name=f"ai_marketing_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="main_download_demo_dashboard",
            )
            st.sidebar.success("Demo export ready.")
        except Exception as exc:
            st.sidebar.error(f"Export failed: {exc}")

    if _can_administer() and st.sidebar.button("Reset Demo Metrics", key="main_reset_demo_metrics"):
        clear_event_log()
        demo_pages.clear_demo_runs()
        demo_pages.clear_model_cost_events()
        log_audit_event(
            action="dashboard_reset_demo_metrics",
            user_id=CURRENT_USER["user_id"],
            role=CURRENT_USER["role"],
        )
        st.sidebar.success("Demo metrics cleared.")
        st.rerun()


# ---------------------------------------------------------------------------
# Screen 1: System Overview
# ---------------------------------------------------------------------------

def render_system_overview():
    st.header("System Overview")

    df = _events_df()
    model_df = _model_events(df)
    tool_df = _tool_events(df)
    error_df = _error_events(df)

    model_calls = len(model_df[model_df["event_type"] == "model_call"]) if not model_df.empty else 0
    model_errors = len(model_df[model_df["event_type"] == "model_error"]) if not model_df.empty else 0
    tool_calls = len(tool_df[tool_df["event_type"] == "tool_call"]) if not tool_df.empty else 0
    tool_errors = len(tool_df[tool_df["event_type"] == "tool_error"]) if not tool_df.empty else 0

    gemini_summary = summarize_cost_evidence([], df.to_dict("records") if not df.empty else [])
    total_tokens = int(gemini_summary["gemini_total_tokens"])
    cached_tokens = int(gemini_summary["gemini_cached_tokens"])
    cache_rate = _safe_pct(cached_tokens, total_tokens)

    overview_cards = [
        ("Gemini Calls", model_calls, "Supervisory model requests", "#38BDF8"),
        (
            "Gemini Errors",
            model_errors,
            "Quota or model runtime failures",
            "#22C55E" if model_errors == 0 else "#EF4444",
        ),
        ("Tool Calls", tool_calls, "Specialist tool executions", "#A78BFA"),
        (
            "Tool Errors",
            tool_errors,
            "Tool wrapper or serving failures",
            "#22C55E" if tool_errors == 0 else "#F97316",
        ),
        ("Gemini Tokens", f"{total_tokens:,}", "Tracked input, output, and cache tokens", "#F59E0B"),
        (
            "Gemini Cost",
            f"${gemini_summary['gemini_token_cost_usd']:.6f}",
            "Estimated API token spend",
            "#14B8A6",
        ),
        ("Cache Hit Rate", f"{cache_rate:.1f}%", "Cached token share", "#EC4899"),
    ]
    columns = st.columns([1, 1, 1, 1, 1.15, 1.15, 1])
    for column, (label, value, hint, accent) in zip(columns, overview_cards):
        with column:
            _overview_card(label, value, hint, accent)

    # Agent health table
    st.subheader("Agent Health")
    if not df.empty and "agent_name" in df.columns:
        health_rows = []
        for agent in df["agent_name"].unique():
            agent_df = df[df["agent_name"] == agent]
            total = len(agent_df)
            errors = len(agent_df[agent_df["event_type"].str.contains("error", na=False)])
            success_rate = _safe_pct(total - errors, total)
            latencies = agent_df["latency_ms"].dropna()
            avg_lat = float(latencies.mean()) if not latencies.empty else 0.0

            if total == 0:
                status = "idle"
            elif success_rate >= 95:
                status = "healthy"
            elif success_rate >= 75:
                status = "degraded"
            else:
                status = "critical"

            health_rows.append({
                "Agent": agent,
                "Status": status,
                "Calls": total,
                "Errors": errors,
                "Success Rate": f"{success_rate:.1f}%",
                "Avg Latency (ms)": f"{avg_lat:.1f}",
            })

        health_df = pd.DataFrame(health_rows)

        def _color_status(val):
            colors = {
                "healthy": "background-color: #d4edda",
                "degraded": "background-color: #fff3cd",
                "critical": "background-color: #f8d7da",
                "idle": "background-color: #e2e3e5",
            }
            return colors.get(val, "")

        styled = health_df.style.map(_color_status, subset=["Status"])
        st.dataframe(styled, **_DF_STRETCH_KW, hide_index=True)
    else:
        st.info("No events recorded yet. Use the Live Agent Tester to generate data.")

    # Recent activity feed
    st.subheader("Recent Activity")
    if not df.empty:
        recent = df.sort_values("timestamp", ascending=False).head(20)
        display_cols = [c for c in ["timestamp", "event_type", "agent_name", "tool_name", "latency_ms", "status", "total_tokens", "error_type"] if c in recent.columns]
        st.dataframe(recent[display_cols], **_DF_STRETCH_KW, hide_index=True)
    else:
        st.info("No events yet.")

    # Error timeline
    if not error_df.empty:
        st.subheader("Error Timeline")
        error_df = error_df.copy()
        error_df["minute"] = error_df["timestamp"].dt.floor("min")
        timeline = error_df.groupby(["minute", "event_type"]).size().reset_index(name="count")
        fig = px.line(
            timeline, x="minute", y="count", color="event_type",
            title="Errors Over Time",
            labels={"minute": "Time", "count": "Error Count"},
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, **_PLOT_STRETCH_KW)


# ---------------------------------------------------------------------------
# Screen 2: Model Performance
# ---------------------------------------------------------------------------

def render_model_performance():
    st.header("Model Performance")

    df = _events_df()
    model_df = _model_events(df)
    success_df = model_df[model_df["event_type"] == "model_call"] if not model_df.empty else pd.DataFrame()

    if success_df.empty:
        st.info("No model call data yet. Run queries via the Live Agent Tester to generate metrics.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Latency Distribution")
        fig = px.histogram(
            success_df, x="latency_ms", nbins=30,
            title="Model Call Latency Distribution",
            labels={"latency_ms": "Latency (ms)"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, **_PLOT_STRETCH_KW)

    with col2:
        st.subheader("Latency by Agent")
        fig = px.box(
            success_df, x="agent_name", y="latency_ms",
            title="Model Latency by Agent",
            labels={"agent_name": "Agent", "latency_ms": "Latency (ms)"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, **_PLOT_STRETCH_KW)

    # Token usage by agent
    st.subheader("Token Usage by Agent")
    token_cols = ["prompt_tokens", "completion_tokens", "cached_tokens"]
    if all(c in success_df.columns for c in token_cols):
        token_agg = success_df.groupby("agent_name")[token_cols].sum().reset_index()
        token_melted = token_agg.melt(
            id_vars="agent_name", value_vars=token_cols,
            var_name="Token Type", value_name="Count",
        )
        fig = px.bar(
            token_melted, x="agent_name", y="Count", color="Token Type",
            title="Token Consumption by Agent",
            barmode="group",
            labels={"agent_name": "Agent"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, **_PLOT_STRETCH_KW)

    # Latency percentiles table
    st.subheader("Latency Percentiles by Agent")
    perc_rows = []
    for agent in success_df["agent_name"].unique():
        lats = success_df[success_df["agent_name"] == agent]["latency_ms"].dropna()
        if lats.empty:
            continue
        perc_rows.append({
            "Agent": agent,
            "Count": len(lats),
            "P50 (ms)": f"{lats.quantile(0.50):.1f}",
            "P95 (ms)": f"{lats.quantile(0.95):.1f}",
            "P99 (ms)": f"{lats.quantile(0.99):.1f}",
            "Mean (ms)": f"{lats.mean():.1f}",
            "Max (ms)": f"{lats.max():.1f}",
        })
    if perc_rows:
        st.dataframe(pd.DataFrame(perc_rows), **_DF_STRETCH_KW, hide_index=True)

    # Token efficiency
    st.subheader("Gemini Token Efficiency")
    monitoring_events = df.to_dict("records") if not df.empty else []
    gemini_summary = summarize_cost_evidence([], monitoring_events)
    total_t = int(gemini_summary["gemini_total_tokens"])
    cached_t = int(gemini_summary["gemini_cached_tokens"])

    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Gemini Total Tokens", f"{total_t:,}")
    e2.metric("Gemini Input Tokens", f"{gemini_summary['gemini_input_tokens']:,}")
    e3.metric("Gemini Output Tokens", f"{gemini_summary['gemini_output_tokens']:,}")
    e4.metric("Gemini Token Cost", f"${gemini_summary['gemini_token_cost_usd']:.6f}")
    e5.metric("Cache Rate", f"{_safe_pct(cached_t, total_t):.1f}%")

    cost_events = load_model_cost_events()
    if cost_events:
        st.subheader("Gemini-Equivalent Cost Evidence")
        cost_summary = summarize_cost_evidence(cost_events, monitoring_events)
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Gemini Model Calls", cost_summary["gemini_model_calls"])
        g2.metric("Gemini Input Tokens", f"{cost_summary['gemini_input_tokens']:,}")
        g3.metric("Gemini Output Tokens", f"{cost_summary['gemini_output_tokens']:,}")
        g4.metric("Gemini Total Tokens", f"{cost_summary['gemini_total_tokens']:,}")
        g5.metric(
            "Gemini Token Cost",
            f"${cost_summary['gemini_token_cost_usd']:.6f}",
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Local Model Calls", cost_summary["local_model_calls"])
        c2.metric("Local Equivalent Tokens", f"{cost_summary['local_total_tokens']:,}")
        c3.metric(
            "Local Equivalent Cost",
            f"${cost_summary['local_gemini_equivalent_cost_usd']:.6f}",
        )
        c4.metric(
            "All-Gemini Baseline",
            f"${cost_summary['all_gemini_baseline_cost_usd']:.6f}",
        )
        c5.metric(
            "API Cost Avoided",
            f"${cost_summary['estimated_api_cost_reduction_usd']:.6f}",
        )
        c6.metric(
            "% Workflow Reduction",
            f"{cost_summary['estimated_api_cost_reduction_pct']:.1f}%",
        )

        story_rows = []
        for story in ("alpha", "beta"):
            story_cost_events = [row for row in cost_events if row.get("story") == story]
            if not story_cost_events:
                continue
            story_summary = summarize_cost_evidence(
                story_cost_events,
                _story_monitoring_events(monitoring_events, story) if story != "unknown" else [],
            )
            story_rows.append({
                "Story": story.title(),
                "Gemini Calls": story_summary["gemini_model_calls"],
                "Gemini Tokens": story_summary["gemini_total_tokens"],
                "Gemini Token Cost": f"${story_summary['gemini_token_cost_usd']:.6f}",
                "Local Calls": story_summary["local_model_calls"],
                "Local Equivalent Tokens": story_summary["local_total_tokens"],
                "Local Equivalent Cost": f"${story_summary['local_gemini_equivalent_cost_usd']:.6f}",
                "All-Gemini Baseline": f"${story_summary['all_gemini_baseline_cost_usd']:.6f}",
                "API Cost Avoided": f"${story_summary['estimated_api_cost_reduction_usd']:.6f}",
                "% Workflow Reduction": f"{story_summary['estimated_api_cost_reduction_pct']:.1f}%",
            })
        if story_rows:
            st.dataframe(pd.DataFrame(story_rows), **_DF_STRETCH_KW, hide_index=True)


# ---------------------------------------------------------------------------
# Screen 3: Tool Performance
# ---------------------------------------------------------------------------

def render_tool_performance():
    st.header("Tool Performance")

    df = _events_df()
    tool_df = _tool_events(df)
    success_tools = tool_df[tool_df["event_type"] == "tool_call"] if not tool_df.empty else pd.DataFrame()
    error_tools = tool_df[tool_df["event_type"] == "tool_error"] if not tool_df.empty else pd.DataFrame()

    if success_tools.empty and error_tools.empty:
        st.info("No tool call data yet. Run queries via the Live Agent Tester to generate metrics.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tool Latency Distribution")
        if not success_tools.empty:
            fig = px.histogram(
                success_tools, x="latency_ms", nbins=30,
                title="Tool Call Latency Distribution",
                labels={"latency_ms": "Latency (ms)"},
                color_discrete_sequence=["#00CC96"],
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, **_PLOT_STRETCH_KW)

    with col2:
        st.subheader("Tool Call Volume")
        if not success_tools.empty and "tool_name" in success_tools.columns:
            vol = success_tools["tool_name"].value_counts().reset_index()
            vol.columns = ["Tool", "Calls"]
            fig = px.bar(
                vol, x="Tool", y="Calls",
                title="Calls per Tool",
                color_discrete_sequence=["#00CC96"],
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, **_PLOT_STRETCH_KW)

    # Success rate per tool
    st.subheader("Tool Success Rate")
    if "tool_name" in tool_df.columns:
        rate_rows = []
        for tool in tool_df["tool_name"].dropna().unique():
            t_df = tool_df[tool_df["tool_name"] == tool]
            total = len(t_df)
            errs = len(t_df[t_df["event_type"] == "tool_error"])
            rate_rows.append({
                "Tool": tool,
                "Total Calls": total,
                "Errors": errs,
                "Success Rate": f"{_safe_pct(total - errs, total):.1f}%",
            })
        if rate_rows:
            rate_df = pd.DataFrame(rate_rows)
            fig = px.bar(
                rate_df, x="Tool", y="Total Calls",
                title="Tool Success / Error Breakdown",
                color_discrete_sequence=["#00CC96"],
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, **_PLOT_STRETCH_KW)
            st.dataframe(rate_df, **_DF_STRETCH_KW, hide_index=True)

    # Slowest tool calls
    st.subheader("Slowest Tool Calls (Top 10)")
    if not success_tools.empty and "latency_ms" in success_tools.columns:
        slowest = success_tools.nlargest(10, "latency_ms")
        display_cols = [c for c in ["timestamp", "tool_name", "agent_name", "latency_ms", "status"] if c in slowest.columns]
        st.dataframe(slowest[display_cols], **_DF_STRETCH_KW, hide_index=True)


# ---------------------------------------------------------------------------
# Screen 4: Log Explorer
# ---------------------------------------------------------------------------

def render_log_explorer():
    st.header("Log Explorer")

    df = _events_df()

    if df.empty:
        st.info("No events recorded yet. Use the Live Agent Tester to generate data.")
        return

    # Filters
    st.subheader("Filters")
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        event_types = df["event_type"].unique().tolist()
        selected_types = st.multiselect("Event Type", event_types, default=event_types)

    with fc2:
        agents = df["agent_name"].dropna().unique().tolist()
        selected_agents = st.multiselect("Agent", agents, default=agents)

    with fc3:
        tools = []
        if "tool_name" in df.columns:
            tools = df["tool_name"].dropna().unique().tolist()
        selected_tools = st.multiselect("Tool", tools, default=tools) if tools else []

    with fc4:
        if "latency_ms" in df.columns:
            lat_vals = df["latency_ms"].dropna()
            if not lat_vals.empty:
                min_lat = float(lat_vals.min())
                max_lat = float(lat_vals.max())
                if min_lat < max_lat:
                    lat_range = st.slider(
                        "Latency Range (ms)",
                        min_value=min_lat,
                        max_value=max_lat,
                        value=(min_lat, max_lat),
                    )
                else:
                    lat_range = (min_lat, max_lat)
            else:
                lat_range = None
        else:
            lat_range = None

    # Text search for error messages
    search_text = st.text_input("Search error messages", "")

    # Apply filters
    filtered = df[df["event_type"].isin(selected_types)]
    filtered = filtered[filtered["agent_name"].isin(selected_agents)]

    if selected_tools and "tool_name" in filtered.columns:
        tool_mask = filtered["tool_name"].isin(selected_tools) | filtered["tool_name"].isna()
        filtered = filtered[tool_mask]

    if lat_range and "latency_ms" in filtered.columns:
        lat_mask = filtered["latency_ms"].isna() | (
            (filtered["latency_ms"] >= lat_range[0]) & (filtered["latency_ms"] <= lat_range[1])
        )
        filtered = filtered[lat_mask]

    if search_text and "error_message" in filtered.columns:
        filtered = filtered[
            filtered["error_message"].fillna("").str.contains(search_text, case=False, na=False)
        ]

    st.subheader(f"Results ({len(filtered)} events)")
    st.dataframe(
        filtered.sort_values("timestamp", ascending=False),
        **_DF_STRETCH_KW,
        hide_index=True,
    )

    # Export
    if not filtered.empty:
        csv = filtered.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"adk_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Screen 5: User Audit
# ---------------------------------------------------------------------------

def render_user_audit():
    st.header("User Audit")
    st.caption("User-scoped login, model, tool, and error audit events.")

    audit_events = get_audit_log()
    if not audit_events:
        st.info("No audit events recorded yet.")
        return

    df = pd.DataFrame(audit_events)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audit Events", len(df))
    c2.metric("Users", df["user_id"].nunique() if "user_id" in df.columns else 0)
    c3.metric("Failures", len(df[df["status"] == "failed"]) if "status" in df.columns else 0)
    c4.metric("Actions", df["action"].nunique() if "action" in df.columns else 0)

    f1, f2, f3 = st.columns(3)
    with f1:
        users = df["user_id"].dropna().unique().tolist() if "user_id" in df.columns else []
        selected_users = st.multiselect("User", users, default=users)
    with f2:
        actions = df["action"].dropna().unique().tolist() if "action" in df.columns else []
        selected_actions = st.multiselect("Action", actions, default=actions)
    with f3:
        statuses = df["status"].dropna().unique().tolist() if "status" in df.columns else []
        selected_statuses = st.multiselect("Status", statuses, default=statuses)

    filtered = df
    if selected_users and "user_id" in filtered.columns:
        filtered = filtered[filtered["user_id"].isin(selected_users)]
    if selected_actions and "action" in filtered.columns:
        filtered = filtered[filtered["action"].isin(selected_actions)]
    if selected_statuses and "status" in filtered.columns:
        filtered = filtered[filtered["status"].isin(selected_statuses)]

    st.subheader("Audit Log")
    st.dataframe(
        filtered.sort_values("timestamp", ascending=False),
        **_DF_STRETCH_KW,
        hide_index=True,
    )

    if not filtered.empty:
        csv = filtered.to_csv(index=False)
        st.download_button(
            "Download Audit CSV",
            csv,
            file_name=f"adk_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    if _can_administer():
        if st.button("Clear Audit Log"):
            clear_audit_log()
            log_audit_event(
                action="dashboard_clear_audit_log",
                user_id=CURRENT_USER["user_id"],
                role=CURRENT_USER["role"],
            )
            st.success("Audit log cleared")
            st.rerun()


# ---------------------------------------------------------------------------
# Screen 6: Live Agent Tester
# ---------------------------------------------------------------------------

@st.cache_resource
def get_dashboard_runner():
    from src.adk_agent.agent import root_agent
    from src.adk_agent.persistent_runtime import create_persistent_runner

    return create_persistent_runner(
        agent=root_agent,
        app_name="ai_marketing_dashboard",
    )


def render_live_agent_tester():
    st.header("Live Agent Tester")
    if not _can_execute():
        st.warning("Your role has read-only dashboard access.")
        return

    stable_session_id = st.session_state.setdefault(
        "dashboard_agent_session_id",
        f"dashboard_{CURRENT_USER['user_id']}",
    )
    st.caption(f"Persistent session: `{stable_session_id}`")
    st.caption(
        "Send queries to the ADK agent system and observe the monitoring "
        "data generated in real time."
    )

    # Preset queries
    presets = {
        "(Custom query)": "",
        "Sentiment — positive review": "Analyze the sentiment of: 'This product is amazing, best purchase ever!'",
        "Sentiment — negative review": "What is the sentiment of: 'Terrible quality, broke after one day.'",
        "Churn — high risk profile": (
            "Is this customer at risk? tenure=2 months, contract=Month-to-month, "
            "internet=Fiber optic, monthly_charges=89.50, no tech support."
        ),
        "Segmentation — champion": (
            "Segment this customer: recency=5 days, frequency=45 orders, "
            "monetary=$12500, average order value=$278."
        ),
        "Support — billing query": (
            "Classify this support query: 'I was charged twice for my subscription "
            "this month. Please refund the duplicate charge.'"
        ),
        "Content strategy — at risk": (
            "What content strategy for: Segment=At Risk, last purchase 90 days ago, "
            "declining engagement, previously high spender."
        ),
        "Recommendation — positive": (
            "Should we recommend this product based on: 'Exceeded all expectations. "
            "Premium build quality, fast delivery, five stars!'"
        ),
    }

    preset_choice = st.selectbox("Preset Queries", list(presets.keys()))
    default_text = presets[preset_choice]

    user_query = st.text_area("Query", value=default_text, height=100)

    if st.button("Run Query", type="primary"):
        if not user_query.strip():
            st.warning("Enter a query first.")
            return

        events_before = len(get_event_log())

        with st.spinner("Running agent query..."):
            try:
                from google.genai import types
                from src.adk_agent.callbacks import get_session_metrics
                from src.adk_agent.persistent_runtime import get_or_create_session

                runner = get_dashboard_runner()

                content = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_query.strip())],
                )

                user_id = CURRENT_USER["user_id"]
                user_role = CURRENT_USER["role"]
                session_id = stable_session_id

                events = []

                async def _run():
                    await get_or_create_session(
                        runner,
                        user_id=user_id,
                        session_id=session_id,
                        role=user_role,
                    )

                    log_audit_event(
                        action="dashboard_query_start",
                        user_id=user_id,
                        role=user_role,
                        metadata={
                            "session_id": session_id,
                            "query_preview": user_query.strip()[:200],
                        },
                    )
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=content,
                    ):
                        events.append(event)

                _loop.run_until_complete(_run())

                # Display response
                st.subheader("Agent Response")
                for event in events:
                    author = getattr(event, "author", "")
                    if hasattr(event, "content") and event.content:
                        for part in getattr(event.content, "parts", []):
                            if hasattr(part, "text") and part.text:
                                st.markdown(f"**[{author}]** {part.text}")

                # Event trace
                st.subheader("Event Trace")
                new_events = get_event_log()[events_before:]
                if new_events:
                    trace_df = pd.DataFrame(new_events)
                    display_cols = [c for c in ["timestamp", "user_id", "role", "session_id", "event_type", "agent_name", "tool_name", "latency_ms", "status", "total_tokens", "error_type"] if c in trace_df.columns]
                    st.dataframe(trace_df[display_cols], **_DF_STRETCH_KW, hide_index=True)

                    # Metrics summary
                    st.subheader("Query Metrics")
                    mc = len([e for e in new_events if e["event_type"] == "model_call"])
                    tc = len([e for e in new_events if e["event_type"] == "tool_call"])
                    me = len([e for e in new_events if e["event_type"] == "model_error"])
                    te = len([e for e in new_events if e["event_type"] == "tool_error"])
                    query_gemini = summarize_cost_evidence([], new_events)
                    ttok = int(query_gemini["gemini_total_tokens"])
                    lats = [e["latency_ms"] for e in new_events if "latency_ms" in e and e["latency_ms"] is not None]
                    total_lat = sum(lats)

                    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                    m1.metric("Gemini Calls", mc)
                    m2.metric("Tool Calls", tc)
                    m3.metric("Errors", me + te)
                    m4.metric("Gemini Tokens", f"{ttok:,}")
                    m5.metric("Gemini Cost", f"${query_gemini['gemini_token_cost_usd']:.6f}")
                    m6.metric("Total Latency", f"{total_lat:.0f} ms")
                    m7.metric("Events", len(new_events))
                else:
                    st.info("No monitoring events captured for this query.")

                st.success("Query completed. Check other dashboard screens for updated metrics.")
                log_audit_event(
                    action="dashboard_query_complete",
                    user_id=user_id,
                    role=user_role,
                    metadata={"session_id": session_id, "events": len(events)},
                )

            except Exception as e:
                log_audit_event(
                    action="dashboard_query_error",
                    user_id=CURRENT_USER["user_id"],
                    role=CURRENT_USER["role"],
                    status="failed",
                    metadata={"error": str(e)[:300]},
                )
                st.error(f"Error running query: {e}")

    # Quick help
    with st.expander("Agent System Info"):
        st.markdown("""
**Available Agents:**
| Agent | Model | Accuracy | Tools |
|-------|-------|----------|-------|
| Sentiment | DistilBERT | 79.8% | analyze_sentiment, analyze_sentiment_batch |
| Churn | XGBoost + Qwen-LoRA | 74.0% / 79.1% | predict_churn_xgboost, predict_churn_llm |
| Segmentation | Qwen-LoRA v2 | 91.2% | segment_customer, segment_customers_batch |
| Support | Qwen-LoRA | 94.7% | classify_support_intent |
| Content | Qwen-LoRA | 98.2% | recommend_content_strategy |
| Recommendation | Qwen-LoRA | 74.0% | recommend_product_action, recommend_product_actions_batch |
        """)


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

if page == "System Overview":
    render_system_overview()
elif page == "Model Performance":
    render_model_performance()
elif page == "Tool Performance":
    render_tool_performance()
elif page == "Log Explorer":
    render_log_explorer()
elif page == "User Audit":
    render_user_audit()
elif page == "Live Agent Tester":
    render_live_agent_tester()
elif page == "Demo Simulation":
    from src.adk_agent import demo as demo_pages

    demo_pages.warn_if_model_server_unhealthy()
    demo_pages.render_demo_simulation()
elif page == "Demo Performance Dashboard":
    from src.adk_agent import demo as demo_pages

    demo_pages.render_performance()
elif page == "Demo Before vs After Impact":
    from src.adk_agent import demo as demo_pages

    demo_pages.render_before_after()
elif page == "Demo Log Explorer":
    from src.adk_agent import demo as demo_pages

    demo_pages.render_log_explorer()
elif page == "Demo System Architecture":
    from src.adk_agent import demo as demo_pages

    demo_pages.render_architecture()

# Auto-refresh
if auto_refresh:
    import time as _time
    _time.sleep(refresh_interval)
    st.rerun()
