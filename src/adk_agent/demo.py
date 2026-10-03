"""Alpha/Beta Story Demo Simulation with Open-Source Data.

Simulates the SYSTEM_STORY_ALPHA_BETA.md scenarios using real open-source
datasets, routing queries through the ADK multi-agent system with full
monitoring.  Captures real-time performance metrics for evaluation.

Datasets:
  - Telco Customer Churn (Kaggle)      → Churn prediction
  - Amazon Food Reviews (Kaggle)       → Sentiment & recommendation
  - UCI Online Retail (UCI ML)         → Customer segmentation
  - IMDB Reviews (Stanford)            → Sentiment analysis
  - Twitter Customer Support (TWCS)    → Support intent classification

Run (two terminals):

    # Terminal 1 — model inference server (loads torch models in separate process)
    python -m src.adk_agent.model_server

    # Terminal 2 — Streamlit demo
    streamlit run src/adk_agent/demo.py
"""

from __future__ import annotations

# Prevent segfault: loky leaks semaphores and the resource_tracker
# unlinks them while still live, crashing the process.  Skipping
# registration means the tracker never learns about them.
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
import os
import sys
import time

import nest_asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
nest_asyncio.apply(loop)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_env_file_fallback(env_path: Path) -> None:
    """Minimal .env loader for environments without python-dotenv."""
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Non-fatal: app will still surface key-status diagnostics.
        pass


# Load .env explicitly for Streamlit runs (env vars are not always auto-loaded).
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    _load_env_file_fallback(PROJECT_ROOT / ".env")

from src.adk_agent.callbacks import clear_event_log, get_event_log
from src.adk_agent.cost_tracking import summarize_cost_evidence
from src.adk_agent.monitoring_store import (
    clear_model_cost_events,
    clear_demo_run,
    clear_demo_runs,
    get_demo_run,
    load_model_cost_events,
    save_demo_run,
)
from src.adk_agent.tools.model_client import model_cost_context, server_healthy

def configure_demo_page() -> None:
    st.set_page_config(
        page_title="Digital Marketing Intelligence System - Demo",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def warn_if_model_server_unhealthy() -> None:
    if not server_healthy():
        st.warning(
            "**Model server not running.** Start it in a separate terminal first:\n\n"
            "```bash\npython -m src.adk_agent.model_server\n```",
            icon="⚠️",
        )


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


def inject_demo_theme() -> None:
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

            .demo-sidebar-brand {
                font-size: 1.18rem;
                font-weight: 850;
                line-height: 1.18;
                color: #F8FAFC;
                padding: 0.35rem 0 0.2rem 0;
            }

            .demo-sidebar-subtitle {
                color: #94A3B8;
                font-size: 0.80rem;
                margin-bottom: 0.7rem;
            }

            .demo-hero {
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 14px;
                padding: 1.1rem 1.2rem;
                margin: 0.1rem 0 1.0rem 0;
                background:
                    linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.74));
                box-shadow: 0 18px 42px rgba(2, 6, 23, 0.28);
            }

            .demo-eyebrow {
                color: #5EEAD4;
                font-size: 0.76rem;
                font-weight: 850;
                text-transform: uppercase;
                letter-spacing: 0;
                margin-bottom: 0.3rem;
            }

            .demo-title {
                color: #F8FAFC;
                font-size: 1.85rem;
                line-height: 1.1;
                font-weight: 850;
                margin-bottom: 0.35rem;
            }

            .demo-subtitle {
                color: #CBD5E1;
                font-size: 0.95rem;
                line-height: 1.45;
                max-width: 980px;
            }

            .demo-section-title {
                color: #F8FAFC;
                font-size: 1.18rem;
                font-weight: 850;
                margin: 1.1rem 0 0.7rem 0;
            }

            .demo-metric-card {
                min-height: 118px;
                border-radius: 12px;
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-top: 4px solid var(--accent);
                background:
                    linear-gradient(135deg, rgba(255, 255, 255, 0.080), rgba(255, 255, 255, 0.025));
                box-shadow: 0 14px 30px rgba(2, 6, 23, 0.25);
                padding: 0.92rem 0.95rem;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                margin-bottom: 0.85rem;
            }

            .demo-metric-label {
                color: #CBD5E1;
                font-size: 0.72rem;
                font-weight: 850;
                text-transform: uppercase;
                letter-spacing: 0;
                margin-bottom: 0.4rem;
            }

            .demo-metric-value {
                color: #F8FAFC;
                font-size: 1.62rem;
                line-height: 1.05;
                font-weight: 850;
                overflow-wrap: anywhere;
            }

            .demo-metric-hint {
                color: #94A3B8;
                font-size: 0.72rem;
                line-height: 1.28;
                margin-top: 0.62rem;
            }

            .demo-callout {
                border-radius: 12px;
                border: 1px solid rgba(45, 212, 191, 0.28);
                background: rgba(13, 148, 136, 0.10);
                padding: 0.85rem 1rem;
                color: #CCFBF1;
                margin: 0.8rem 0;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 12px 28px rgba(2, 6, 23, 0.18);
            }

            div[data-testid="stTabs"] button {
                font-weight: 800;
            }

            div[data-testid="stTabs"] div[role="tablist"] {
                gap: 0.35rem;
                padding: 0 0 0.65rem 0;
                margin-bottom: 1.1rem;
                border-bottom: 1px solid rgba(148, 163, 184, 0.26);
            }

            div[data-testid="stTabs"] div[role="tabpanel"] {
                padding-top: 0.35rem;
            }

            div[data-testid="stTabs"] div[role="tab"] {
                border-radius: 10px 10px 0 0;
                padding: 0.65rem 0.9rem;
            }

            div[data-testid="stTabs"] div[role="tab"][aria-selected="true"] {
                background: rgba(20, 184, 166, 0.12);
                border-bottom-color: #EF4444;
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


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _page_header(title: str, subtitle: str, eyebrow: str = "Digital Marketing Intelligence System") -> None:
    st.markdown(
        f"""
        <div class="demo-hero">
            <div class="demo-eyebrow">{_escape(eyebrow)}</div>
            <div class="demo-title">{_escape(title)}</div>
            <div class="demo-subtitle">{_escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_title(title: str) -> None:
    st.markdown(
        f'<div class="demo-section-title">{_escape(title)}</div>',
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: Any, hint: str = "", accent: str = "#38BDF8") -> None:
    st.markdown(
        f"""
        <div class="demo-metric-card" style="--accent: {_escape(accent)};">
            <div>
                <div class="demo-metric-label">{_escape(label)}</div>
                <div class="demo-metric-value">{_escape(value)}</div>
            </div>
            <div class="demo-metric-hint">{_escape(hint)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_grid(cards: List[Dict[str, Any]], columns_per_row: int = 4) -> None:
    for start in range(0, len(cards), columns_per_row):
        row = cards[start:start + columns_per_row]
        columns = st.columns(len(row))
        for column, card in zip(columns, row):
            with column:
                _metric_card(
                    card["label"],
                    card["value"],
                    card.get("hint", ""),
                    card.get("accent", "#38BDF8"),
                )


def _avg_latency_label(df: pd.DataFrame) -> str:
    if df.empty or "latency_ms" not in df.columns:
        return "N/A"
    latencies = df["latency_ms"].dropna()
    return f"{latencies.mean():.0f} ms" if not latencies.empty else "N/A"


def _format_label(label: str) -> str:
    return label.replace("_", " ").title()


def _render_cached_table(rows: Any) -> None:
    if isinstance(rows, list):
        if rows:
            st.dataframe(pd.DataFrame(rows), **_DF_STRETCH_KW, hide_index=True)
        return
    if isinstance(rows, dict):
        if rows:
            st.dataframe(pd.DataFrame([rows]), **_DF_STRETCH_KW, hide_index=True)
        return
    st.write(rows)


def _render_cached_output_group(label: str, rows: Any) -> None:
    if not rows:
        return

    st.markdown(f"**{_format_label(label)}**")
    if isinstance(rows, dict):
        rendered_nested = False
        for sub_label, sub_rows in rows.items():
            if not sub_rows:
                continue
            rendered_nested = True
            st.caption(_format_label(sub_label))
            _render_cached_table(sub_rows)
        if not rendered_nested:
            st.caption("No cached rows available.")
        return

    _render_cached_table(rows)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data"


@st.cache_data
def load_churn_data() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "processed" / "churn" / "churn_test.csv")


@st.cache_data
def load_rfm_data() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "processed" / "segmentation" / "rfm_features.csv")


@st.cache_data
def load_sentiment_data() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / "processed" / "sentiment" / "sentiment_sample.csv")


@st.cache_data
def load_reviews_data() -> pd.DataFrame:
    p = DATA_ROOT / "raw" / "recommendation" / "Reviews.csv"
    if p.exists():
        return pd.read_csv(p, nrows=500)
    return pd.DataFrame()


@st.cache_data
def load_support_data() -> pd.DataFrame:
    p = DATA_ROOT / "raw" / "support" / "twcs" / "twcs.csv"
    if p.exists():
        df = pd.read_csv(p)
        inbound = df[df["inbound"] == True].copy()
        inbound = inbound[inbound["text"].str.len() > 20]
        return inbound.head(200)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Query builders — transform dataset rows → natural-language ADK queries
# ---------------------------------------------------------------------------

CONTRACT_MAP = {0: "Month-to-month", 1: "One year", 2: "Two year"}
INTERNET_MAP = {0: "DSL", 1: "Fiber optic", 2: "No"}
PAYMENT_MAP = {
    0: "Electronic check", 1: "Mailed check",
    2: "Bank transfer (automatic)", 3: "Credit card (automatic)",
}
YES_NO = {0: "No", 1: "Yes"}


def _safe_map(val, mapping, default="Unknown"):
    if isinstance(val, str):
        return val
    return mapping.get(int(val), default) if pd.notna(val) else default


def churn_query(row) -> str:
    contract = _safe_map(row.get("Contract"), CONTRACT_MAP)
    internet = _safe_map(row.get("InternetService"), INTERNET_MAP)
    payment = _safe_map(row.get("PaymentMethod"), PAYMENT_MAP)
    tenure = int(row.get("tenure", 0))
    monthly = float(row.get("MonthlyCharges", 0))
    total = float(row.get("TotalCharges", 0))
    return (
        f"Predict churn risk for this customer: tenure={tenure} months, "
        f"monthly_charges={monthly:.2f}, total_charges={total:.2f}, "
        f"contract={contract}, internet_service={internet}, "
        f"payment_method={payment}"
    )


def segmentation_query(row) -> str:
    recency = int(row.get("Recency", 0))
    frequency = int(row.get("Frequency", 0))
    monetary = float(row.get("Monetary", 0))
    return (
        f"Segment this customer: recency={recency} days, "
        f"frequency={frequency} orders, monetary=${monetary:.2f}"
    )


def sentiment_query(text: str) -> str:
    clean = text.strip()[:300]
    return f"Analyze the sentiment of this review: '{clean}'"


def recommendation_query(text: str) -> str:
    clean = text.strip()[:300]
    return f"Should we recommend this product based on this review: '{clean}'"


def support_query(text: str) -> str:
    clean = text.strip()[:300]
    return f"Classify the intent of this support query: '{clean}'"


def content_query(segment: str, context: str = "") -> str:
    return (
        f"Recommend a content strategy for this customer profile: "
        f"Segment={segment}, {context}"
    )


# ---------------------------------------------------------------------------
# ADK runner
# ---------------------------------------------------------------------------

@st.cache_resource
def get_runner():
    from src.adk_agent.agent import root_agent
    from src.adk_agent.persistent_runtime import create_persistent_runner

    return create_persistent_runner(agent=root_agent, app_name="ai_marketing_demo")


def _is_placeholder_key(value: str | None) -> bool:
    if not value:
        return False
    v = value.strip().lower()
    return (
        v == "your_google_api_key_here"
        or v == "your_gemini_api_key_here"
        or v.startswith("your_")
        or v.endswith("_here")
    )


def _configured_gemini_key() -> tuple[str | None, str | None]:
    for var_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        raw = os.getenv(var_name)
        if raw and raw.strip() and not _is_placeholder_key(raw):
            return raw.strip(), var_name
    return None, None


def _gemini_config_error_message() -> str:
    return (
        "Configuration error: Gemini API key is missing or placeholder. "
        "Set a real key in `.env` using `GOOGLE_API_KEY=<your_real_key>` "
        "or `GEMINI_API_KEY=<your_real_key>`, then restart Streamlit."
    )


def run_single_query(
    query: str,
    session_id: str = "demo",
    story: str | None = None,
    workflow_step: str | None = None,
) -> str:
    """Run a query through the ADK agent and return the text response."""
    from google.genai import types

    key, key_name = _configured_gemini_key()
    if not key:
        return _gemini_config_error_message()

    runner = get_runner()
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)],
    )

    responses = []

    async def _run():
        from src.adk_agent.persistent_runtime import get_or_create_session

        await get_or_create_session(
            runner,
            user_id="demo_user",
            session_id=session_id,
            role="demo_user",
        )

        async for event in runner.run_async(
            user_id="demo_user",
            session_id=session_id,
            new_message=content,
        ):
            if hasattr(event, "content") and event.content:
                for part in getattr(event.content, "parts", []):
                    if hasattr(part, "text") and part.text:
                        responses.append(part.text)

    try:
        with model_cost_context(story=story, workflow_step=workflow_step):
            asyncio.run(_run())
    except ValueError as exc:
        if "No API key was provided" in str(exc):
            return _gemini_config_error_message()
        raise

    return "\n".join(responses) if responses else "(No response)"


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def run_alpha_campaign(container):
    """Alpha Step A: Daily Campaign Pipeline — Segmentation + Content Strategy."""
    rfm = load_rfm_data()
    segments = ["Champions", "Loyal", "At Risk", "Needs Attention"]
    samples = []
    for seg in segments:
        seg_rows = rfm[rfm["Segment"] == seg]
        if not seg_rows.empty:
            samples.append(seg_rows.sample(1).iloc[0])
    if len(samples) < 4:
        samples = [rfm.sample(1).iloc[0] for _ in range(4)]

    container.markdown("**Step A: Daily Campaign Pipeline** — UCI Online Retail data")
    progress = container.progress(0, text="Starting segmentation pipeline...")
    results = []
    total = len(samples) * 2

    for i, row in enumerate(samples):
        progress.progress((i * 2) / total, text=f"Segmenting customer {i+1}/{len(samples)}...")
        q = segmentation_query(row)
        resp = run_single_query(
            q,
            session_id=f"alpha_campaign_{i}",
            story="alpha",
            workflow_step="campaign_segmentation",
        )
        seg_result = resp

        progress.progress((i * 2 + 1) / total, text=f"Content strategy for customer {i+1}...")
        cq = content_query(
            segment=row.get("Segment", "Unknown"),
            context=f"recency={int(row.get('Recency',0))} days, frequency={int(row.get('Frequency',0))} orders, monetary=${float(row.get('Monetary',0)):.0f}"
        )
        content_resp = run_single_query(
            cq,
            session_id=f"alpha_campaign_{i}",
            story="alpha",
            workflow_step="campaign_content_strategy",
        )

        results.append({
            "Customer": str(row.get("CustomerID", f"#{i+1}")),
            "RFM": f"R={int(row.get('Recency',0))}, F={int(row.get('Frequency',0))}, M=${float(row.get('Monetary',0)):.0f}",
            "Dataset Segment": row.get("Segment", "?"),
            "Segmentation Response": seg_result[:200],
            "Content Strategy": content_resp[:200],
        })

    progress.progress(1.0, text="Campaign pipeline complete.")
    container.dataframe(pd.DataFrame(results), **_DF_STRETCH_KW, hide_index=True)
    return results


def run_alpha_monitoring(container):
    """Alpha Step B: Midday Risk Monitoring — Churn + Sentiment in parallel."""
    churn_df = load_churn_data()
    reviews_df = load_reviews_data()
    churn_results = []
    sent_results = []

    container.markdown("**Step B: Risk Monitoring** — Telco Churn (Kaggle) + Amazon Reviews (Kaggle)")

    col1, col2 = container.columns(2)

    with col1:
        st.markdown("**Churn Risk Scan**")
        churn_sample = churn_df.sample(5, random_state=42)
        progress_c = st.progress(0, text="Scanning churn risk...")
        for i, (_, row) in enumerate(churn_sample.iterrows()):
            progress_c.progress((i + 1) / 5, text=f"Customer {i+1}/5...")
            q = churn_query(row)
            resp = run_single_query(
                q,
                session_id=f"alpha_churn_{i}",
                story="alpha",
                workflow_step="risk_churn_scan",
            )
            actual = "Yes" if row.get("Churn", 0) == 1 else "No"
            churn_results.append({
                "Tenure": int(row.get("tenure", 0)),
                "Monthly": f"${float(row.get('MonthlyCharges', 0)):.2f}",
                "Actual Churn": actual,
                "AI Prediction": resp[:150],
            })
        st.dataframe(pd.DataFrame(churn_results), **_DF_STRETCH_KW, hide_index=True)

    with col2:
        st.markdown("**Sentiment Watch**")
        if not reviews_df.empty:
            review_sample = reviews_df.sample(min(5, len(reviews_df)), random_state=42)
            progress_s = st.progress(0, text="Analyzing sentiment...")
            for i, (_, row) in enumerate(review_sample.iterrows()):
                progress_s.progress((i + 1) / 5, text=f"Review {i+1}/5...")
                text = str(row.get("Text", row.get("Summary", "")))[:200]
                q = sentiment_query(text)
                resp = run_single_query(
                    q,
                    session_id=f"alpha_sentiment_{i}",
                    story="alpha",
                    workflow_step="risk_sentiment_watch",
                )
                sent_results.append({
                    "Review": text[:80] + "...",
                    "Score": row.get("Score", "?"),
                    "AI Analysis": resp[:150],
                })
            st.dataframe(pd.DataFrame(sent_results), **_DF_STRETCH_KW, hide_index=True)
        else:
            st.info("Reviews dataset not found.")

    return {
        "churn_risk_scan": churn_results,
        "sentiment_watch": sent_results,
    }


def run_alpha_recommendation(container):
    """Alpha Step C: Product Recommendation Assessment."""
    reviews_df = load_reviews_data()
    container.markdown("**Step C: Product Recommendation** — Amazon Food Reviews (Kaggle)")

    if reviews_df.empty:
        container.info("Reviews dataset not found.")
        return []

    score_groups = {1: "1-star", 3: "3-star", 5: "5-star"}
    samples = []
    for score in [1, 3, 5]:
        grp = reviews_df[reviews_df["Score"] == score]
        if not grp.empty:
            samples.append(grp.sample(1).iloc[0])

    remaining = 5 - len(samples)
    if remaining > 0:
        samples.extend([reviews_df.sample(1).iloc[0] for _ in range(remaining)])

    progress = container.progress(0, text="Assessing product recommendations...")
    results = []
    for i, row in enumerate(samples):
        progress.progress((i + 1) / len(samples), text=f"Review {i+1}/{len(samples)}...")
        text = str(row.get("Text", ""))[:250]
        q = recommendation_query(text)
        resp = run_single_query(
            q,
            session_id=f"alpha_rec_{i}",
            story="alpha",
            workflow_step="product_recommendation",
        )
        results.append({
            "Product": str(row.get("ProductId", "?"))[:15],
            "Score": row.get("Score", "?"),
            "Review": text[:80] + "...",
            "AI Recommendation": resp[:200],
        })

    progress.progress(1.0, text="Recommendation assessment complete.")
    container.dataframe(pd.DataFrame(results), **_DF_STRETCH_KW, hide_index=True)
    return results


def run_beta_retention(container):
    """Beta Step A: Retention & Churn Prevention for high-risk customers."""
    churn_df = load_churn_data()
    container.markdown("**Step A: Retention Intervention** — Telco Churn (Kaggle)")

    churners = churn_df[churn_df["Churn"] == 1] if "Churn" in churn_df.columns else churn_df.head(5)
    if churners.empty:
        churners = churn_df.head(5)

    sample = churners.sample(min(5, len(churners)), random_state=99)
    progress = container.progress(0, text="Identifying at-risk customers...")
    results = []

    for i, (_, row) in enumerate(sample.iterrows()):
        progress.progress((i * 2) / 10, text=f"Churn prediction for customer {i+1}/5...")
        q = churn_query(row)
        churn_resp = run_single_query(
            q,
            session_id=f"beta_retention_{i}",
            story="beta",
            workflow_step="retention_churn_prediction",
        )

        progress.progress((i * 2 + 1) / 10, text=f"Retention strategy for customer {i+1}/5...")
        cq = content_query(
            segment="At Risk",
            context=f"tenure={int(row.get('tenure',0))} months, "
                    f"monthly_charges=${float(row.get('MonthlyCharges',0)):.2f}, "
                    f"high churn risk, needs immediate retention action"
        )
        content_resp = run_single_query(
            cq,
            session_id=f"beta_retention_{i}",
            story="beta",
            workflow_step="retention_content_strategy",
        )

        results.append({
            "Tenure": int(row.get("tenure", 0)),
            "Monthly": f"${float(row.get('MonthlyCharges', 0)):.2f}",
            "Churn Analysis": churn_resp[:150],
            "Retention Strategy": content_resp[:150],
        })

    progress.progress(1.0, text="Retention workflow complete.")
    container.dataframe(pd.DataFrame(results), **_DF_STRETCH_KW, hide_index=True)
    return results


def run_beta_support(container):
    """Beta Step B: Support Escalation Routing."""
    support_df = load_support_data()
    container.markdown("**Step B: Support Routing** — Twitter Customer Support (TWCS)")

    if support_df.empty:
        container.info("Support dataset not found.")
        return []

    sample = support_df.sample(min(5, len(support_df)), random_state=77)
    progress = container.progress(0, text="Classifying support tickets...")
    results = []

    for i, (_, row) in enumerate(sample.iterrows()):
        progress.progress((i + 1) / 5, text=f"Ticket {i+1}/5...")
        text = str(row.get("text", ""))
        text = text.lstrip("@").split(" ", 1)[-1] if text.startswith("@") else text
        q = support_query(text[:250])
        resp = run_single_query(
            q,
            session_id=f"beta_support_{i}",
            story="beta",
            workflow_step="support_routing",
        )
        results.append({
            "Ticket": text[:100] + "...",
            "AI Classification": resp[:200],
        })

    progress.progress(1.0, text="Support routing complete.")
    container.dataframe(pd.DataFrame(results), **_DF_STRETCH_KW, hide_index=True)
    return results


def run_beta_crisis(container):
    """Beta Step C: Crisis Monitoring — Sentiment Watch."""
    sent_df = load_sentiment_data()
    container.markdown("**Step C: Crisis Monitoring** — IMDB Reviews (Stanford)")

    if sent_df.empty:
        container.info("Sentiment dataset not found.")
        return []

    neg = sent_df[sent_df["Sentiment"] == "negative"]
    pos = sent_df[sent_df["Sentiment"] == "positive"]
    sample_texts = []
    for df_part in [neg.head(5), pos.head(5)]:
        for _, row in df_part.iterrows():
            sample_texts.append(str(row.get("CleanText", ""))[:200])

    sample_texts = [t for t in sample_texts if len(t) > 10][:10]

    if not sample_texts:
        container.info("No usable review texts found.")
        return []

    reviews_str = " | ".join([f"'{t[:80]}'" for t in sample_texts[:5]])
    q = f"Analyze sentiment for these reviews: {reviews_str}"

    progress = container.progress(0, text="Running batch sentiment analysis...")
    resp = run_single_query(
        q,
        session_id="beta_crisis",
        story="beta",
        workflow_step="crisis_sentiment_monitoring",
    )
    progress.progress(1.0, text="Sentiment analysis complete.")

    container.markdown("**Batch Analysis Result:**")
    container.markdown(resp[:1000])

    return sample_texts


# ---------------------------------------------------------------------------
# Tab 1: Demo Simulation
# ---------------------------------------------------------------------------

def render_demo_simulation():
    _page_header(
        "Demo Simulation",
        "Run the product-company and service-company scenarios, then review persisted outputs after Streamlit reruns or page navigation.",
    )
    render_story_status_cards()

    story = st.session_state.get("demo_story", "Alpha Story")
    alpha_run = get_demo_run("alpha")
    beta_run = get_demo_run("beta")

    if story in ("Alpha Story", "Full Demo"):
        _section_title("Story Alpha - E-Commerce Product Company")
        st.caption("UCI Online Retail + Kaggle Telco Churn + Amazon Reviews")

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            if st.button("Run Alpha Story", key="run_alpha", type="primary"):
                events_before = len(get_event_log())
                t0 = time.time()

                c1 = st.container()
                campaign_results = run_alpha_campaign(c1)
                st.markdown("---")

                c2 = st.container()
                monitoring_results = run_alpha_monitoring(c2)
                st.markdown("---")

                c3 = st.container()
                recommendation_results = run_alpha_recommendation(c3)

                elapsed = time.time() - t0
                new_events = len(get_event_log()) - events_before
                save_demo_run(
                    "alpha",
                    elapsed,
                    new_events,
                    outputs={
                        "campaign": campaign_results,
                        "monitoring": monitoring_results,
                        "recommendations": recommendation_results,
                    },
                )
                st.success(f"Alpha story complete: {elapsed:.1f}s elapsed, {new_events} monitoring events captured.")
                st.rerun()  # Refresh dashboard to show updated metrics in other tabs
        
        with col_btn2:
            if alpha_run and st.button("Clear Alpha", key="clear_alpha"):
                clear_demo_run("alpha")
                clear_model_cost_events("alpha")
                st.rerun()

        # Show cached results
        alpha_run = get_demo_run("alpha")
        if alpha_run:
            with st.expander("Alpha Story Results (cached)", expanded=True):
                st.info(
                    "Last run: "
                    f"{alpha_run['elapsed_seconds']:.1f}s elapsed, "
                    f"{alpha_run['event_count']} monitoring events captured "
                    f"at {alpha_run['completed_at']}"
                )
                for label, rows in alpha_run.get("outputs", {}).items():
                    _render_cached_output_group(label, rows)

    if story in ("Alpha Story", "Full Demo"):
        st.markdown("---")

    if story in ("Beta Story", "Full Demo"):
        _section_title("Story Beta - Subscription Service Provider")
        st.caption("Kaggle Telco Churn + Twitter Support (TWCS) + IMDB Reviews")
        st.dataframe(
            pd.DataFrame([
                {
                    "Step": "A. Retention",
                    "Workflow": "Chained",
                    "Agents Used": "Churn → Content",
                    "Dataset": "Telco Churn",
                },
                {
                    "Step": "B. Support Routing",
                    "Workflow": "Single",
                    "Agents Used": "Support",
                    "Dataset": "Twitter TWCS",
                },
                {
                    "Step": "C. Crisis Monitoring",
                    "Workflow": "Batch",
                    "Agents Used": "Sentiment",
                    "Dataset": "IMDB Reviews",
                },
            ]),
            **_DF_STRETCH_KW,
            hide_index=True,
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            if st.button("Run Beta Story", key="run_beta", type="primary"):
                events_before = len(get_event_log())
                t0 = time.time()

                c1 = st.container()
                retention_results = run_beta_retention(c1)
                st.markdown("---")

                c2 = st.container()
                support_results = run_beta_support(c2)
                st.markdown("---")

                c3 = st.container()
                crisis_results = run_beta_crisis(c3)

                elapsed = time.time() - t0
                new_events = len(get_event_log()) - events_before
                save_demo_run(
                    "beta",
                    elapsed,
                    new_events,
                    outputs={
                        "retention": retention_results,
                        "support": support_results,
                        "crisis": crisis_results,
                    },
                )
                st.success(f"Beta story complete: {elapsed:.1f}s elapsed, {new_events} monitoring events captured.")
                st.rerun()  # Refresh dashboard to show updated metrics in other tabs
        
        with col_btn2:
            if beta_run and st.button("Clear Beta", key="clear_beta"):
                clear_demo_run("beta")
                clear_model_cost_events("beta")
                st.rerun()

        # Show cached results
        beta_run = get_demo_run("beta")
        if beta_run:
            with st.expander("Beta Story Results (cached)", expanded=True):
                st.info(
                    "Last run: "
                    f"{beta_run['elapsed_seconds']:.1f}s elapsed, "
                    f"{beta_run['event_count']} monitoring events captured "
                    f"at {beta_run['completed_at']}"
                )
                for label, rows in beta_run.get("outputs", {}).items():
                    _render_cached_output_group(label, rows)


# ---------------------------------------------------------------------------
# Tab 2: Performance Dashboard
# ---------------------------------------------------------------------------

def _safe_pct(num: int, den: int) -> float:
    return (num / den * 100) if den > 0 else 0.0


def _run_summary(run: Dict[str, Any] | None) -> tuple[str, str]:
    if not run:
        return "Not run", "Awaiting scenario execution"
    return (
        "Completed",
        f"{float(run.get('elapsed_seconds', 0.0)):.1f}s, {int(run.get('event_count', 0))} events",
    )


def render_story_status_cards() -> None:
    alpha_run = get_demo_run("alpha")
    beta_run = get_demo_run("beta")
    alpha_status, alpha_hint = _run_summary(alpha_run)
    beta_status, beta_hint = _run_summary(beta_run)
    cards = [
        {
            "label": "Alpha Product Story",
            "value": alpha_status,
            "hint": alpha_hint,
            "accent": "#22C55E" if alpha_run else "#F59E0B",
        },
        {
            "label": "Beta Service Story",
            "value": beta_status,
            "hint": beta_hint,
            "accent": "#22C55E" if beta_run else "#F59E0B",
        },
        {
            "label": "Evidence Store",
            "value": "SQLite",
            "hint": "Demo outputs persist through page changes",
            "accent": "#38BDF8",
        },
    ]
    _metric_grid(cards, columns_per_row=3)


def _story_monitoring_events(events: List[Dict[str, Any]], story: str) -> List[Dict[str, Any]]:
    prefix = f"{story}_"
    return [
        event
        for event in events
        if str(event.get("session_id", "")).startswith(prefix)
    ]


def render_system_result_overview(events: List[Dict[str, Any]], cost_events: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(events) if events else pd.DataFrame()
    model_ok = df[df["event_type"] == "model_call"] if not df.empty and "event_type" in df.columns else pd.DataFrame()
    total_errors = (
        len(df[df["event_type"].str.contains("error", na=False)])
        if not df.empty and "event_type" in df.columns
        else 0
    )
    error_rate = _safe_pct(total_errors, len(events))
    cost_summary = summarize_cost_evidence(cost_events, events)
    completed_runs = sum(1 for story in ("alpha", "beta") if get_demo_run(story))
    cards = [
        {
            "label": "Demo Runs",
            "value": f"{completed_runs}/2",
            "hint": "Product and service scenarios persisted",
            "accent": "#22C55E" if completed_runs == 2 else "#F59E0B",
        },
        {
            "label": "Monitoring Events",
            "value": f"{len(events):,}",
            "hint": "Model/tool callback evidence",
            "accent": "#38BDF8",
        },
        {
            "label": "Gemini Tokens",
            "value": f"{int(cost_summary['gemini_total_tokens']):,}",
            "hint": "Input plus output orchestration tokens",
            "accent": "#A78BFA",
        },
        {
            "label": "Local Model Calls",
            "value": f"{int(cost_summary['local_model_calls']):,}",
            "hint": "Specialist calls served by FastAPI",
            "accent": "#2DD4BF",
        },
        {
            "label": "API Cost Avoided",
            "value": f"${float(cost_summary['estimated_api_cost_reduction_usd']):.6f}",
            "hint": "Gemini-equivalent local specialist cost",
            "accent": "#F97316",
        },
        {
            "label": "Error Rate",
            "value": f"{error_rate:.1f}%",
            "hint": "Model and tool error events",
            "accent": "#EF4444" if error_rate else "#22C55E",
        },
        {
            "label": "Avg Model Latency",
            "value": _avg_latency_label(model_ok),
            "hint": "Mean latency across model-call events",
            "accent": "#38BDF8",
        },
        {
            "label": "Evidence Status",
            "value": "Reviewable",
            "hint": "SQLite cost/demo rows and callback logs",
            "accent": "#22C55E",
        },
    ]
    _metric_grid(cards, columns_per_row=4)


def render_performance():
    _page_header(
        "Demo Performance Dashboard",
        "System result overview for Alpha/Beta simulations, runtime traces, token usage, local-model cost evidence, and agent health.",
    )

    events = get_event_log()
    cost_events = load_model_cost_events()
    _section_title("System Result Overview")
    render_system_result_overview(events, cost_events)

    if not events:
        st.info("No events yet. Run a demo scenario first.")
        return

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    model_ok = df[df["event_type"] == "model_call"] if "event_type" in df.columns else pd.DataFrame()
    model_err = df[df["event_type"] == "model_error"] if "event_type" in df.columns else pd.DataFrame()
    tool_ok = df[df["event_type"] == "tool_call"] if "event_type" in df.columns else pd.DataFrame()
    tool_err = df[df["event_type"] == "tool_error"] if "event_type" in df.columns else pd.DataFrame()

    gemini_summary = summarize_cost_evidence([], events)
    total_tokens = int(gemini_summary["gemini_total_tokens"])
    cached = int(gemini_summary["gemini_cached_tokens"])

    _section_title("Live Performance Metrics")
    _metric_grid([
        {"label": "Gemini Calls", "value": gemini_summary["gemini_model_calls"], "hint": "ADK model-call events", "accent": "#38BDF8"},
        {"label": "Gemini Errors", "value": len(model_err), "hint": "Model error callbacks", "accent": "#EF4444" if len(model_err) else "#22C55E"},
        {"label": "Tool Calls", "value": len(tool_ok), "hint": "Specialist tool calls", "accent": "#2DD4BF"},
        {"label": "Tool Errors", "value": len(tool_err), "hint": "Tool error callbacks", "accent": "#EF4444" if len(tool_err) else "#22C55E"},
        {"label": "Gemini Tokens", "value": f"{total_tokens:,}", "hint": "Input and output tokens", "accent": "#A78BFA"},
        {"label": "Gemini Cost", "value": f"${gemini_summary['gemini_token_cost_usd']:.6f}", "hint": "Configured token pricing", "accent": "#F97316"},
        {"label": "Cache Rate", "value": f"{_safe_pct(cached, total_tokens):.1f}%", "hint": "Cached token share", "accent": "#22C55E"},
    ], columns_per_row=4)

    if cost_events:
        cost_summary = summarize_cost_evidence(cost_events, events)
        _section_title("Gemini-Equivalent Cost Evidence")
        _metric_grid([
            {"label": "Gemini Model Calls", "value": cost_summary["gemini_model_calls"], "hint": "Supervisory model calls", "accent": "#38BDF8"},
            {"label": "Gemini Input Tokens", "value": f"{cost_summary['gemini_input_tokens']:,}", "hint": "Prompt tokens", "accent": "#A78BFA"},
            {"label": "Gemini Output Tokens", "value": f"{cost_summary['gemini_output_tokens']:,}", "hint": "Completion tokens", "accent": "#A78BFA"},
            {"label": "Gemini Token Cost", "value": f"${cost_summary['gemini_token_cost_usd']:.6f}", "hint": "Observed Gemini-priced cost", "accent": "#F97316"},
            {"label": "Local Model Calls", "value": cost_summary["local_model_calls"], "hint": "FastAPI specialist calls", "accent": "#2DD4BF"},
            {"label": "Local Equivalent Cost", "value": f"${cost_summary['local_gemini_equivalent_cost_usd']:.6f}", "hint": "Equivalent Gemini cost", "accent": "#2DD4BF"},
            {"label": "All-Gemini Baseline", "value": f"${cost_summary['all_gemini_baseline_cost_usd']:.6f}", "hint": "Gemini plus local-equivalent cost", "accent": "#F59E0B"},
            {"label": "Workflow Reduction", "value": f"{cost_summary['estimated_api_cost_reduction_pct']:.1f}%", "hint": "Avoided cost share", "accent": "#22C55E"},
        ], columns_per_row=4)

        story_rows = []
        for story in ("alpha", "beta"):
            story_cost_events = [row for row in cost_events if row.get("story") == story]
            if not story_cost_events:
                continue
            story_summary = summarize_cost_evidence(
                story_cost_events,
                _story_monitoring_events(events, story) if story != "unknown" else [],
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

    col1, col2 = st.columns(2)

    with col1:
        if not model_ok.empty and "latency_ms" in model_ok.columns:
            fig = px.box(
                model_ok, x="agent_name", y="latency_ms",
                title="Model Latency by Agent",
                labels={"agent_name": "Agent", "latency_ms": "Latency (ms)"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, **_PLOT_STRETCH_KW)

    with col2:
        if not model_ok.empty and "total_tokens" in model_ok.columns:
            token_agg = model_ok.groupby("agent_name")[["prompt_tokens", "completion_tokens"]].sum().reset_index()
            token_melted = token_agg.melt(id_vars="agent_name", var_name="Type", value_name="Tokens")
            fig = px.bar(
                token_melted, x="agent_name", y="Tokens", color="Type",
                title="Token Usage by Agent", barmode="group",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, **_PLOT_STRETCH_KW)

    _section_title("Agent Health")
    if "agent_name" in df.columns:
        rows = []
        for agent in df["agent_name"].unique():
            adf = df[df["agent_name"] == agent]
            total = len(adf)
            errs = len(adf[adf["event_type"].str.contains("error", na=False)])
            sr = _safe_pct(total - errs, total)
            lats = adf["latency_ms"].dropna()
            rows.append({
                "Agent": agent,
                "Status": "healthy" if sr >= 95 else ("degraded" if sr >= 75 else "critical"),
                "Calls": total,
                "Errors": errs,
                "Success Rate": f"{sr:.1f}%",
                "Avg Latency": f"{lats.mean():.0f} ms" if not lats.empty else "-",
            })
        st.dataframe(pd.DataFrame(rows), **_DF_STRETCH_KW, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Before vs After Impact
# ---------------------------------------------------------------------------

def render_before_after():
    _page_header(
        "Before vs After Impact",
        "Comparison of manual marketing operations with the AI-assisted workflow produced by the demo scenarios.",
    )

    events = get_event_log()
    df = pd.DataFrame(events) if events else pd.DataFrame()

    model_ok = df[df["event_type"] == "model_call"] if not df.empty and "event_type" in df.columns else pd.DataFrame()
    avg_latency = f"{model_ok['latency_ms'].mean():.0f} ms" if not model_ok.empty and "latency_ms" in model_ok.columns else "N/A (run demo first)"

    impact_data = [
        {
            "Marketing Function": "Customer Segmentation",
            "Before AI": "Manual spreadsheet analysis, weekly updates",
            "After AI": "Real-time LLM classification (Qwen2.5 LoRA)",
            "Manual Time": "2-4 hours",
            "AI Time": avg_latency,
            "Model Accuracy": "91.2%",
            "Dataset": "UCI Online Retail (4,338 customers)",
        },
        {
            "Marketing Function": "Churn Prediction",
            "Before AI": "Post-analysis after a customer leaves",
            "After AI": "Proactive risk scoring (XGBoost + LLM)",
            "Manual Time": "Days (after churn)",
            "AI Time": avg_latency,
            "Model Accuracy": "74.0% / ROC-AUC 0.846",
            "Dataset": "Telco Customer Churn (7,043 customers)",
        },
        {
            "Marketing Function": "Sentiment Analysis",
            "Before AI": "Manual review reading, sample-based",
            "After AI": "Automated 3-class classification (DistilBERT)",
            "Manual Time": "1-2 hours per batch",
            "AI Time": avg_latency,
            "Model Accuracy": "79.8% / ROC-AUC 0.93",
            "Dataset": "Amazon Reviews + IMDB (550K+ reviews)",
        },
        {
            "Marketing Function": "Content Strategy",
            "Before AI": "One-size-fits-all campaigns",
            "After AI": "Segment-aware strategy recommendation (Qwen2.5 LoRA)",
            "Manual Time": "1-2 days per segment",
            "AI Time": avg_latency,
            "Model Accuracy": "98.2%",
            "Dataset": "Derived from segmentation + churn signals",
        },
        {
            "Marketing Function": "Support Routing",
            "Before AI": "Manual ticket triage, inconsistent routing",
            "After AI": "5-class intent classification (Qwen2.5 LoRA)",
            "Manual Time": "5-10 min per ticket",
            "AI Time": avg_latency,
            "Model Accuracy": "94.7%",
            "Dataset": "Twitter Customer Support (3K+ tickets)",
        },
        {
            "Marketing Function": "Product Recommendation",
            "Before AI": "Generic promotions, no review-based filtering",
            "After AI": "Review-driven recommend/consider/reject (Qwen2.5 LoRA)",
            "Manual Time": "Manual curation",
            "AI Time": avg_latency,
            "Model Accuracy": "74.0% / F1 0.73",
            "Dataset": "Amazon Food Reviews (500K+ reviews)",
        },
    ]

    st.dataframe(
        pd.DataFrame(impact_data),
        **_DF_STRETCH_KW,
        hide_index=True,
        column_config={
            "Marketing Function": st.column_config.TextColumn(width="medium"),
            "Before AI": st.column_config.TextColumn(width="large"),
            "After AI": st.column_config.TextColumn(width="large"),
        },
    )

    _section_title("Aggregate Impact Metrics")

    total_events = len(events)
    gemini_summary = summarize_cost_evidence([], events)
    total_tokens = int(gemini_summary["gemini_total_tokens"])
    total_errors = len(df[df["event_type"].str.contains("error", na=False)]) if not df.empty and "event_type" in df.columns else 0
    error_rate = _safe_pct(total_errors, total_events)

    _metric_grid([
        {"label": "Total AI Decisions", "value": total_events, "hint": "Model and tool invocations", "accent": "#38BDF8"},
        {"label": "Gemini Tokens", "value": f"{total_tokens:,}", "hint": "Processed orchestration tokens", "accent": "#A78BFA"},
        {"label": "Gemini Cost", "value": f"${gemini_summary['gemini_token_cost_usd']:.6f}", "hint": "Observed token cost", "accent": "#F97316"},
        {"label": "Error Rate", "value": f"{error_rate:.1f}%", "hint": "Runtime event error share", "accent": "#EF4444" if error_rate else "#22C55E"},
        {"label": "Avg Decision Latency", "value": avg_latency, "hint": "Mean model-call latency", "accent": "#2DD4BF"},
    ], columns_per_row=5)

    st.markdown("""
**Key Improvements:**
- **Speed**: From hours/days of manual work to millisecond-level AI decisions
- **Coverage**: From sample-based analysis to processing every customer and review
- **Consistency**: From subjective human judgment to reproducible model outputs
- **Proactivity**: From reactive post-mortem to real-time risk monitoring
- **Personalisation**: From generic campaigns to segment-native content strategies
    """)


# ---------------------------------------------------------------------------
# Tab 4: Log Explorer
# ---------------------------------------------------------------------------

def render_log_explorer():
    _page_header(
        "Demo Log Explorer",
        "Filter runtime callback records by event type, agent, and error text for traceable demo evidence.",
    )

    events = get_event_log()
    if not events:
        st.info("No events yet. Run a demo scenario to generate data.")
        return

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        types_avail = df["event_type"].unique().tolist()
        sel_types = st.multiselect("Event Type", types_avail, default=types_avail)
    with fc2:
        agents_avail = df["agent_name"].dropna().unique().tolist()
        sel_agents = st.multiselect("Agent", agents_avail, default=agents_avail)
    with fc3:
        search = st.text_input("Search (error messages)")

    filtered = df[df["event_type"].isin(sel_types) & df["agent_name"].isin(sel_agents)]
    if search and "error_message" in filtered.columns:
        filtered = filtered[filtered["error_message"].fillna("").str.contains(search, case=False)]

    st.caption(f"Showing {len(filtered)} of {len(df)} events")
    st.dataframe(
        filtered.sort_values("timestamp", ascending=False),
        **_DF_STRETCH_KW,
        hide_index=True,
    )

    if not filtered.empty:
        csv = filtered.to_csv(index=False)
        st.download_button(
            "Export CSV", csv,
            file_name=f"demo_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Tab 5: System Architecture
# ---------------------------------------------------------------------------

def render_architecture():
    _page_header(
        "Demo System Architecture",
        "High-level view of the Streamlit demo, ADK orchestration, FastAPI model server, specialist models, and evidence stores.",
    )

    st.markdown("""
```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                     Streamlit Dashboard (demo.py)                         │
  │                    UI only — no torch model loading                       │
  └────────────────────────────────┬──────────────────────────────────────────┘
                                   │ ADK Runner
                        ┌──────────▼──────────────────┐
                        │   AI Marketing Orchestrator  │
                        │    (Gemini 3.1 Flash Lite)  │
                        └──────────┬──────────────────┘
                                   │ routes to
          ┌────────────┬───────────┼───────────┬────────────┬──────────────┐
          ▼            ▼           ▼           ▼            ▼              ▼
    ┌──────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────┐
    │Sentiment │ │  Churn   │ │Segment │ │ Support │ │ Content  │ │Recommendation│
    │  Agent   │ │  Agent   │ │ Agent  │ │  Agent  │ │  Agent   │ │    Agent     │
    └─────┬────┘ └─────┬────┘ └───┬────┘ └────┬────┘ └─────┬────┘ └──────┬───────┘
          │            │          │           │            │             │
          ▼            ▼          ▼           ▼            ▼             ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │              Tool Functions → HTTP (httpx) → Model Server                   │
    └──────────────────────────────────┬──────────────────────────────────────────┘
                                       │ port 8100
    ┌──────────────────────────────────▼──────────────────────────────────────────┐
    │              FastAPI Model Server (model_server.py)                         │
    │    Separate process — loads all torch models on Apple Silicon GPU (MPS)     │
    │                                                                             │
    │  DistilBERT  │ XGBoost │ Qwen2.5-LoRA x5 (churn, segment, support,          │
    │   79.8%      │  74.0%  │              content, recommendation)              │
    └─────────────────────────────────────────────────────────────────────────────┘
```
    """)

    _section_title("Open-Source Datasets")
    st.dataframe(pd.DataFrame([
        {"Dataset": "UCI Online Retail", "Source": "UCI ML Repository", "Records": "4,338 customers", "Agent": "Segmentation", "Features": "RFM (Recency, Frequency, Monetary)"},
        {"Dataset": "Telco Customer Churn", "Source": "Kaggle (blastchar)", "Records": "7,043 customers", "Agent": "Churn XGBoost", "Features": "19 telecom features + 17 engineered"},
        {"Dataset": "Amazon Food Reviews", "Source": "Kaggle", "Records": "500K+ reviews", "Agent": "Sentiment, Recommendation", "Features": "Review text, score, helpfulness"},
        {"Dataset": "IMDB Movie Reviews", "Source": "Stanford AI Lab", "Records": "50K reviews", "Agent": "Sentiment", "Features": "Review text, binary sentiment"},
        {"Dataset": "Twitter Customer Support", "Source": "TWCS (Kaggle)", "Records": "3K+ tickets", "Agent": "Support Intent", "Features": "Customer query text, author, timestamp"},
    ]), **_DF_STRETCH_KW, hide_index=True)

    _section_title("Alpha & Beta Story Mapping")
    st.markdown("""
| Story | Step | Workflow | Agents Used | Dataset |
|-------|------|----------|-------------|---------|
| **Alpha** | A. Campaign Pipeline | Sequential | Segmentation → Content | UCI Online Retail |
| **Alpha** | B. Risk Monitoring | Parallel | Churn + Sentiment | Telco Churn + Amazon Reviews |
| **Alpha** | C. Recommendation | Single | Recommendation | Amazon Food Reviews |
| **Beta** | A. Retention | Chained | Churn → Content | Telco Churn |
| **Beta** | **B. Support Routing** | **Single** | **Support** | **Twitter TWCS** |
| **Beta** | C. Crisis Monitoring | Batch | Sentiment | IMDB Reviews |
    """)

    _section_title("Fine-Tuned Models")
    st.dataframe(pd.DataFrame([
        {"Model": "DistilBERT Sentiment", "Base": "distilbert-base-uncased", "Method": "Full fine-tune", "Accuracy": "79.8%", "ROC-AUC": "0.93", "Classes": "negative, neutral, positive"},
        {"Model": "XGBoost Churn", "Base": "XGBoost", "Method": "Optuna HPO", "Accuracy": "74.0%", "ROC-AUC": "0.846", "Classes": "HIGH_RISK, LOW_RISK"},
        {"Model": "Churn LLM", "Base": "Qwen2.5-0.5B", "Method": "QLoRA", "Accuracy": "79.1%", "ROC-AUC": "-", "Classes": "HIGH_RISK, LOW_RISK"},
        {"Model": "Segmentation LLM", "Base": "Qwen2.5-0.5B", "Method": "QLoRA", "Accuracy": "91.2%", "ROC-AUC": "-", "Classes": "Champions, Loyal, At Risk, Needs Attention"},
        {"Model": "Support LLM", "Base": "Qwen2.5-0.5B", "Method": "QLoRA", "Accuracy": "94.7%", "ROC-AUC": "-", "Classes": "ACCOUNT, BILLING, DELIVERY, GENERAL, TECHNICAL"},
        {"Model": "Content LLM", "Base": "Qwen2.5-0.5B", "Method": "QLoRA", "Accuracy": "98.2%", "ROC-AUC": "-", "Classes": "DISCOUNT_REENGAGE, EDUCATIONAL_NURTURE, LOYALTY_UPSELL"},
        {"Model": "Recommendation LLM", "Base": "Qwen2.5-0.5B", "Method": "QLoRA", "Accuracy": "74.0%", "ROC-AUC": "-", "Classes": "RECOMMEND, CONSIDER, DO_NOT_RECOMMEND"},
    ]), **_DF_STRETCH_KW, hide_index=True)


# ---------------------------------------------------------------------------
# HTML Export Function
# ---------------------------------------------------------------------------

def export_dashboard_html() -> str:
    """Generate complete dashboard as static HTML with all pages."""
    
    # Gather all data
    events = get_event_log()
    df_events = pd.DataFrame(events) if events else pd.DataFrame()
    
    # Calculate metrics - initialize all variables first
    model_ok = pd.DataFrame()
    model_err = pd.DataFrame()
    tool_ok = pd.DataFrame()
    tool_err = pd.DataFrame()
    total_tokens = 0
    cached = 0
    gemini_cost = 0.0
    avg_latency = "N/A"
    cache_rate = 0
    gemini_summary = summarize_cost_evidence([], events)
    total_tokens = int(gemini_summary["gemini_total_tokens"])
    cached = int(gemini_summary["gemini_cached_tokens"])
    gemini_cost = float(gemini_summary["gemini_token_cost_usd"])
    cache_rate = (cached / total_tokens * 100) if total_tokens > 0 else 0
    
    if not df_events.empty:
        df_events["timestamp"] = pd.to_datetime(df_events["timestamp"])
        model_ok = df_events[df_events["event_type"] == "model_call"] if "event_type" in df_events.columns else pd.DataFrame()
        model_err = df_events[df_events["event_type"] == "model_error"] if "event_type" in df_events.columns else pd.DataFrame()
        tool_ok = df_events[df_events["event_type"] == "tool_call"] if "event_type" in df_events.columns else pd.DataFrame()
        tool_err = df_events[df_events["event_type"] == "tool_error"] if "event_type" in df_events.columns else pd.DataFrame()
        
        avg_latency = f"{model_ok['latency_ms'].mean():.0f} ms" if not model_ok.empty and "latency_ms" in model_ok.columns else "N/A"
    
    # Build charts for performance dashboard
    chart_latency_html = ""
    chart_tokens_html = ""
    
    if not model_ok.empty:
        if "latency_ms" in model_ok.columns and "agent_name" in model_ok.columns:
            fig_latency = px.box(model_ok, x="agent_name", y="latency_ms", title="Model Latency by Agent")
            chart_latency_html = pio.to_html(fig_latency, full_html=False, include_plotlyjs=False)
        
        if "total_tokens" in model_ok.columns and "agent_name" in model_ok.columns:
            token_agg = model_ok.groupby("agent_name")[["prompt_tokens", "completion_tokens"]].sum().reset_index()
            token_melted = token_agg.melt(id_vars="agent_name", var_name="Type", value_name="Tokens")
            fig_tokens = px.bar(token_melted, x="agent_name", y="Tokens", color="Type", title="Token Usage by Agent")
            chart_tokens_html = pio.to_html(fig_tokens, full_html=False, include_plotlyjs=False)
    
    # Agent health data
    agent_health_html = "<tr><td colspan='6' style='text-align:center; padding:20px;'>No agent data available</td></tr>"
    if "agent_name" in df_events.columns:
        rows = []
        for agent in df_events["agent_name"].unique():
            adf = df_events[df_events["agent_name"] == agent]
            total = len(adf)
            errs = len(adf[adf["event_type"].str.contains("error", na=False)])
            sr = ((total - errs) / total * 100) if total > 0 else 0
            lats = adf["latency_ms"].dropna()
            avg_lat = f"{lats.mean():.0f} ms" if not lats.empty else "-"
            status_color = "#00d4ff" if sr >= 95 else ("#ff9500" if sr >= 75 else "#ff4444")
            rows.append(f"""
            <tr>
                <td>{agent}</td>
                <td style='color: {status_color}; font-weight: bold;'>{'healthy' if sr >= 95 else ('degraded' if sr >= 75 else 'critical')}</td>
                <td>{total}</td>
                <td>{errs}</td>
                <td>{sr:.1f}%</td>
                <td>{avg_lat}</td>
            </tr>
            """)
        agent_health_html = "".join(rows)
    
    # Impact data
    impact_data = [
        {"function": "Customer Segmentation", "before": "Manual spreadsheet analysis, weekly updates", "after": "Real-time LLM classification"},
        {"function": "Churn Prediction", "before": "Post-analysis after a customer leaves", "after": "Proactive risk scoring (XGBoost + LLM)"},
        {"function": "Sentiment Analysis", "before": "Manual review reading, sample-based", "after": "Automated 3-class classification"},
        {"function": "Content Strategy", "before": "One-size-fits-all campaigns", "after": "Segment-aware strategy recommendation"},
        {"function": "Support Routing", "before": "Manual ticket triage", "after": "5-class intent classification"},
        {"function": "Product Recommendation", "before": "Generic promotions", "after": "Review-driven filtering"},
    ]
    
    impact_rows = "".join([f"""
    <tr>
        <td><strong>{item['function']}</strong></td>
        <td>{item['before']}</td>
        <td>{item['after']}</td>
    </tr>
    """ for item in impact_data])
    
    # Event log table - full explorer version
    event_table_html = ""
    if not df_events.empty:
        all_events = df_events.sort_values("timestamp", ascending=False)
        event_rows = []
        for _, event in all_events.iterrows():
            ts = event.get("timestamp", "")
            et = event.get("event_type", "-")
            agent = event.get("agent_name", "-")
            latency = f"{event.get('latency_ms', '-'):.0f}" if pd.notna(event.get('latency_ms')) else "-"
            tokens = f"{event.get('total_tokens', '-')}" if pd.notna(event.get('total_tokens')) else "-"
            msg = str(event.get("error_message", ""))[:80] if pd.notna(event.get("error_message")) else ""
            event_rows.append(f"""
            <tr>
                <td style='font-size: 0.85em;'>{ts}</td>
                <td><span style='font-size: 0.85em; padding: 2px 6px; border-radius: 3px; background: #1e2130;'>{et}</span></td>
                <td>{agent}</td>
                <td style='text-align: center;'>{latency}</td>
                <td style='text-align: center;'>{tokens}</td>
                <td style='color: #ff9500; font-size: 0.85em;'>{msg}</td>
            </tr>
            """)
        event_table_html = "".join(event_rows)
    else:
        event_table_html = "<tr><td colspan='6' style='text-align:center; padding:20px; color: #8b949e;'>No events recorded</td></tr>"
    
    # Build complete HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Marketing Demo — Complete Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
                color: #c9d1d9;
                padding: 40px 20px;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            
            header {{
                text-align: center;
                margin-bottom: 50px;
                border-bottom: 2px solid #30363d;
                padding-bottom: 30px;
            }}
            
            h1 {{
                font-size: 2.5em;
                color: #00d4ff;
                margin-bottom: 10px;
                text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
            }}
            
            .subtitle {{
                color: #8b949e;
                font-size: 1.1em;
            }}
            
            h2 {{
                color: #00d4ff;
                font-size: 1.8em;
                margin-top: 50px;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 1px solid #30363d;
            }}
            
            h3 {{
                color: #58a6ff;
                font-size: 1.3em;
                margin-top: 30px;
                margin-bottom: 15px;
            }}
            
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            
            .metric {{
                background: linear-gradient(135deg, #1e2130 0%, #22262e 100%);
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                border: 1px solid #30363d;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            
            .metric:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 20px rgba(0, 212, 255, 0.1);
            }}
            
            .metric-value {{
                font-size: 2.2em;
                color: #00d4ff;
                font-weight: bold;
                margin-bottom: 8px;
            }}
            
            .metric-label {{
                color: #8b949e;
                font-size: 0.95em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .card {{
                background: linear-gradient(135deg, #1e2130 0%, #22262e 100%);
                border-radius: 12px;
                padding: 30px;
                margin: 25px 0;
                border: 1px solid #30363d;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: #0d1117;
                border-radius: 8px;
                overflow: hidden;
            }}
            
            th {{
                background: #161b22;
                color: #00d4ff;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid #30363d;
            }}
            
            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #30363d;
            }}
            
            tr:hover {{
                background: #161b22;
            }}
            
            .chart {{
                margin: 30px 0;
                padding: 20px;
                background: #0d1117;
                border-radius: 8px;
                border: 1px solid #30363d;
            }}
            
            .badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                margin: 0 5px;
            }}
            
            .badge-healthy {{
                background: rgba(0, 212, 255, 0.2);
                color: #00d4ff;
            }}
            
            .badge-degraded {{
                background: rgba(255, 149, 0, 0.2);
                color: #ff9500;
            }}
            
            .badge-critical {{
                background: rgba(255, 68, 68, 0.2);
                color: #ff4444;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 1px solid #30363d;
                color: #8b949e;
                font-size: 0.9em;
            }}
            
            .generated-at {{
                color: #58a6ff;
                font-weight: 500;
            }}
            
            @media (max-width: 768px) {{
                h1 {{ font-size: 1.8em; }}
                h2 {{ font-size: 1.3em; }}
                .metrics {{ grid-template-columns: 1fr; }}
                table {{ font-size: 0.9em; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <header>
                <h1>📊 AI Marketing Demo — Complete Dashboard</h1>
                <p class="subtitle">Comprehensive Performance & Impact Analysis</p>
                <p class="generated-at">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </header>
            
            <!-- Key Metrics -->
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{len(model_ok)}</div>
                    <div class="metric-label">Gemini Calls</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(df_events[df_events['event_type'] == 'model_error']) if not df_events.empty and 'event_type' in df_events.columns else 0}</div>
                    <div class="metric-label">Gemini Errors</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(tool_ok)}</div>
                    <div class="metric-label">Tool Calls</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(df_events[df_events['event_type'] == 'tool_error']) if not df_events.empty and 'event_type' in df_events.columns else 0}</div>
                    <div class="metric-label">Tool Errors</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_tokens:,}</div>
                    <div class="metric-label">Gemini Tokens</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${gemini_cost:.6f}</div>
                    <div class="metric-label">Gemini Cost</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{cache_rate:.1f}%</div>
                    <div class="metric-label">Cache Rate</div>
                </div>
            </div>
            
            <!-- Performance Section -->
            <h2>📈 Performance Metrics</h2>
            <div class="card">
                <h3>Average Latency</h3>
                <p style="font-size: 1.8em; color: #00d4ff; margin: 15px 0;">{avg_latency}</p>
                <p style="color: #8b949e;">Time from query to response across all model calls</p>
            </div>
            
            {f'<div class="card chart">{chart_latency_html}</div>' if chart_latency_html else ''}
            {f'<div class="card chart">{chart_tokens_html}</div>' if chart_tokens_html else ''}
            
            <!-- Agent Health -->
            <h2>🏥 Agent Health Status</h2>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th>Agent</th>
                            <th>Status</th>
                            <th>Total Calls</th>
                            <th>Errors</th>
                            <th>Success Rate</th>
                            <th>Avg Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {agent_health_html}
                    </tbody>
                </table>
            </div>
            
            <!-- Impact Analysis -->
            <h2>💡 AI Impact on Digital Marketing</h2>
            <div class="card">
                <h3>Before vs After Comparison</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Marketing Function</th>
                            <th style="width: 40%;">Before AI</th>
                            <th style="width: 40%;">After AI</th>
                        </tr>
                    </thead>
                    <tbody>
                        {impact_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h3>Key Improvements</h3>
                <ul style="margin-left: 20px; color: #8b949e;">
                    <li><strong style="color: #00d4ff;">Speed:</strong> From hours/days of manual work to millisecond-level AI decisions</li>
                    <li><strong style="color: #00d4ff;">Coverage:</strong> From sample-based analysis to processing every customer and review</li>
                    <li><strong style="color: #00d4ff;">Consistency:</strong> From subjective human judgment to reproducible model outputs</li>
                    <li><strong style="color: #00d4ff;">Proactivity:</strong> From reactive post-mortem to real-time risk monitoring</li>
                    <li><strong style="color: #00d4ff;">Personalization:</strong> From generic campaigns to segment-native content strategies</li>
                </ul>
            </div>
            
            <!-- Event Log -->
            <h2>📋 Recent Event Log (Last 50 Events)</h2>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Event Type</th>
                            <th>Agent</th>
                            <th>Latency</th>
                            <th>Error Message</th>
                        </tr>
                    </thead>
                    <tbody>
                        {event_table_html}
                    </tbody>
                </table>
            </div>
            
            <!-- System Info -->
            <h2>🏗️ System Information</h2>
            <div class="card">
                <h3>Datasets Used</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Source</th>
                            <th>Size</th>
                            <th>Agent</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>UCI Online Retail</td>
                            <td>UCI ML Repository</td>
                            <td>4,338 customers</td>
                            <td>Segmentation</td>
                        </tr>
                        <tr>
                            <td>Telco Customer Churn</td>
                            <td>Kaggle (blastchar)</td>
                            <td>7,043 customers</td>
                            <td>Churn XGBoost</td>
                        </tr>
                        <tr>
                            <td>Amazon Food Reviews</td>
                            <td>Kaggle</td>
                            <td>500K+ reviews</td>
                            <td>Sentiment, Recommendation</td>
                        </tr>
                        <tr>
                            <td>IMDB Movie Reviews</td>
                            <td>Stanford AI Lab</td>
                            <td>50K reviews</td>
                            <td>Sentiment</td>
                        </tr>
                        <tr>
                            <td>Twitter Customer Support</td>
                            <td>TWCS (Kaggle)</td>
                            <td>3K+ tickets</td>
                            <td>Support Intent</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h3>Fine-Tuned Models</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Base</th>
                            <th>Method</th>
                            <th>Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>DistilBERT Sentiment</td>
                            <td>distilbert-base-uncased</td>
                            <td>Full fine-tune</td>
                            <td>79.8%</td>
                        </tr>
                        <tr>
                            <td>XGBoost Churn</td>
                            <td>XGBoost</td>
                            <td>Optuna HPO</td>
                            <td>74.0%</td>
                        </tr>
                        <tr>
                            <td>Churn LLM (Qwen2.5)</td>
                            <td>Qwen2.5-0.5B</td>
                            <td>QLoRA</td>
                            <td>79.1%</td>
                        </tr>
                        <tr>
                            <td>Segmentation LLM (Qwen2.5)</td>
                            <td>Qwen2.5-0.5B</td>
                            <td>QLoRA</td>
                            <td>91.2%</td>
                        </tr>
                        <tr>
                            <td>Support LLM (Qwen2.5)</td>
                            <td>Qwen2.5-0.5B</td>
                            <td>QLoRA</td>
                            <td>94.7%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- System Architecture -->
            <h2>🏗️ System Architecture & Story Mapping</h2>
            <div class="card">
                <h3>Alpha & Beta Story Workflow</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Story</th>
                            <th>Step</th>
                            <th>Workflow</th>
                            <th>Agents Used</th>
                            <th>Dataset</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Alpha</strong></td>
                            <td>A. Campaign Pipeline</td>
                            <td>Sequential</td>
                            <td>Segmentation → Content</td>
                            <td>UCI Online Retail</td>
                        </tr>
                        <tr>
                            <td><strong>Alpha</strong></td>
                            <td>B. Risk Monitoring</td>
                            <td>Parallel</td>
                            <td>Churn + Sentiment</td>
                            <td>Telco Churn + Amazon Reviews</td>
                        </tr>
                        <tr>
                            <td><strong>Alpha</strong></td>
                            <td>C. Recommendation</td>
                            <td>Single</td>
                            <td>Recommendation</td>
                            <td>Amazon Food Reviews</td>
                        </tr>
                        <tr>
                            <td><strong>Beta</strong></td>
                            <td>A. Retention</td>
                            <td>Chained</td>
                            <td>Churn → Content</td>
                            <td>Telco Churn</td>
                        </tr>
                        <tr>
                            <td><strong>Beta</strong></td>
                            <td><strong>B. Support Routing</strong></td>
                            <td><strong>Single</strong></td>
                            <td><strong>Support</strong></td>
                            <td><strong>Twitter TWCS</strong></td>
                        </tr>
                        <tr>
                            <td><strong>Beta</strong></td>
                            <td>C. Crisis Monitoring</td>
                            <td>Batch</td>
                            <td>Sentiment</td>
                            <td>IMDB Reviews</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h3>System Architecture Overview</h3>
                <pre style="background: #0d1117; padding: 20px; border-radius: 8px; overflow-x: auto; color: #58a6ff; font-size: 0.85em; line-height: 1.4;">
┌───────────────────────────────────────────────────────────────────────────┐ 
│                     Streamlit Dashboard (demo.py)                         │
│                    UI only — no torch model loading                       │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │ ADK Runner
                      ┌──────────▼──────────────────┐
                      │   AI Marketing Orchestrator │
                      │    (Gemini 3.1 Flash Lite)  │
                      └──────────┬──────────────────┘
                                 │ routes to
        ┌────────────┬───────────┼───────────┬────────────┬──────────────┐
        ▼            ▼           ▼           ▼            ▼              ▼
  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────┐
  │Sentiment │ │  Churn   │ │Segment │ │ Support │ │ Content  │ │Recommendation│
  │  Agent   │ │  Agent   │ │ Agent  │ │  Agent  │ │  Agent   │ │    Agent     │
  └─────┬────┘ └─────┬────┘ └───┬────┘ └────┬────┘ └─────┬────┘ └──────┬───────┘
        │            │          │           │            │             │
        ▼            ▼          ▼           ▼            ▼             ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │              Tool Functions → HTTP (httpx) → Model Server                   │
  └──────────────────────────────────┬──────────────────────────────────────────┘
                                     │ port 8100
  ┌──────────────────────────────────▼──────────────────────────────────────────┐
  │              FastAPI Model Server (model_server.py)                         │
  │    Separate process — loads all torch models on Apple Silicon GPU (MPS)     │
  │                                                                             │
  │  DistilBERT  │ XGBoost │ Qwen2.5-LoRA x5 (churn, segment, support,          │
  │   79.8%      │  74.0%  │              content, recommendation)              │
  └─────────────────────────────────────────────────────────────────────────────┘
                </pre>
            </div>
            
            <!-- Complete Event Log Explorer -->
            <h2>📋 Complete Event Log Explorer</h2>
            <div class="card">
                <p style="color: #8b949e; margin-bottom: 20px;">All {len(df_events)} events from the current session, sorted by recency</p>
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Event Type</th>
                            <th>Agent</th>
                            <th>Latency (ms)</th>
                            <th>Tokens</th>
                            <th>Error Message</th>
                        </tr>
                    </thead>
                    <tbody>
                        {event_table_html}
                    </tbody>
                </table>
            </div>
            
            <!-- Footer -->
            <footer class="footer">
                <p>📊 AI Marketing Demo Dashboard — Powered by Google ADK + Fine-tuned Models</p>
                <p>Open-source datasets: UCI ML, Kaggle, Stanford, TWCS</p>
                <p style="margin-top: 15px; color: #30363d;">Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S UTC')}</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    return html


def render_demo_sidebar() -> None:
    st.sidebar.markdown(
        """
        <div class="demo-sidebar-brand">Digital Marketing<br>Intelligence System</div>
        <div class="demo-sidebar-subtitle">Alpha/Beta simulation dashboard</div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    demo_story = st.sidebar.radio(
        "Select Story",
        ["Alpha Story", "Beta Story", "Full Demo"],
        index=0,
    )
    st.session_state["demo_story"] = demo_story

    st.sidebar.markdown("---")
    event_count = len(get_event_log())
    st.sidebar.metric("Monitoring Events", event_count)

    if st.sidebar.button("Export Dashboard as HTML", key="export_dashboard"):
        try:
            html_content = export_dashboard_html()
            st.sidebar.download_button(
                label="Download dashboard.html",
                data=html_content,
                file_name=f"ai_marketing_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_dashboard",
            )
            st.sidebar.success("Dashboard ready to download.")
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")

    st.sidebar.markdown("---")

    if st.sidebar.button("Reset All Metrics"):
        clear_event_log()
        clear_demo_runs()
        clear_model_cost_events()
        st.sidebar.success("Event log, demo results, and cost evidence cleared.")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Google ADK + locally served specialist models")
    st.sidebar.caption("Open-source datasets: UCI, Kaggle, Stanford, TWCS")

    key, key_name = _configured_gemini_key()
    if key:
        st.sidebar.success(f"Gemini key loaded from `{key_name}`.")
    else:
        st.sidebar.error(
            "Gemini key not configured. Update `.env` and restart Streamlit."
        )


def render_demo_tabs() -> None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Demo Simulation",
        "Performance Dashboard",
        "Before vs After Impact",
        "Log Explorer",
        "System Architecture",
    ])

    with tab1:
        render_demo_simulation()
    with tab2:
        render_performance()
    with tab3:
        render_before_after()
    with tab4:
        render_log_explorer()
    with tab5:
        render_architecture()


def main() -> None:
    configure_demo_page()
    inject_demo_theme()
    warn_if_model_server_unhealthy()
    render_demo_sidebar()
    render_demo_tabs()


if __name__ == "__main__":
    main()
