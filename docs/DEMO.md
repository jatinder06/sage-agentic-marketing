# How-To Guide

This guide explains how to run and verify the ADK-based Digital Marketing Intelligence System from the repository root.

## 1. Prerequisites

- Python 3.11+ virtual environment or conda environment
- Project dependencies installed from `requirements.txt`
- Local model artefacts under `artifacts/` for full model inference
- Optional Google Gemini credentials if live ADK/Gemini orchestration is used

The dashboard can open without a production identity provider. It uses local demo credentials by default, or credentials supplied through `ADK_DASHBOARD_USERS`.

## 2. Install Dependencies

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Download the datasets and model artifacts from the shared Google Drive folder:

```bash
./scripts/setup_artifacts.sh
```

If you use conda:

```bash
conda activate agentic
uv pip install -r requirements.txt
```

## 3. Configure Environment Variables

Minimum optional configuration:

```bash
export MODEL_SERVER_URL="http://localhost:8100"
export ADK_DASHBOARD_USERS="admin:<change-me>:data_scientist,marketing:<change-me>:marketing_manager,viewer:<change-me>:viewer"
```

Gemini-backed ADK calls require one of:

```bash
export GOOGLE_API_KEY="<your-key>"
# or
export GEMINI_API_KEY="<your-key>"
```

Useful runtime controls:

```bash
export GEMINI_RETRY_ATTEMPTS="5"
export GEMINI_RETRY_BACKOFF_FACTOR="2"
export GEMINI_MIN_CALL_INTERVAL_SECONDS="4"
export GEMINI_COST_BASIS_MODEL="gemini-3.1-flash-lite"
export ADK_LOG_LEVEL="INFO"
```

Do not commit real API keys or production credentials.

## 4. Start the Model Server

The model server loads local DistilBERT, XGBoost, and Qwen LoRA artefacts outside the Streamlit process.

```bash
python -m src.adk_agent.model_server
```

Default URL:

```text
http://localhost:8100
```

Health check:

```bash
curl http://localhost:8100/health
```

The server exposes prediction capabilities for:

- sentiment
- structured churn
- text-based churn
- segmentation
- support intent
- content strategy
- recommendation action

## 5. Start the Main Dashboard

Open a second terminal from the repository root:

```bash
streamlit run src/adk_agent/dashboard.py
```

The dashboard provides:

- System Overview
- Model Performance
- Tool Performance
- Log Explorer
- User Audit
- Live Agent Tester
- Demo Simulation
- Demo Performance Dashboard
- Demo Before vs After Impact
- Demo Log Explorer
- Demo System Architecture

Default local demo users are available if `ADK_DASHBOARD_USERS` is not set. They are intended only for local demonstration; set `ADK_DASHBOARD_USERS` before sharing the dashboard.

| Username | Password | Role |
| --- | --- | --- |
| `admin` | local demo password | Data Scientist |
| `marketing` | local demo password | Marketing Manager |
| `viewer` | local demo password | Viewer |

These credentials are for local demonstration only.

## 6. Run the Standalone Demo

The standalone demo is useful for presenting the Alpha and Beta customer-experience scenarios without the main dashboard shell.

```bash
streamlit run src/adk_agent/demo.py
```

Alpha represents a product-company scenario using segmentation, sentiment/churn monitoring, and recommendation evidence.

Beta represents a service-company scenario using churn retention, support routing, and crisis sentiment monitoring.

Demo outputs persist in SQLite so results remain visible after page navigation or Streamlit reruns.

## 7. Run a CLI Query

The CLI runner sends a single prompt through the ADK root agent and records audit/monitoring events.

```bash
python -m src.adk_agent.run "Analyze the sentiment of: The product quality is excellent."
```

Optional telemetry:

```bash
python -m src.adk_agent.run --telemetry "Segment this customer: recency=5, frequency=8, monetary=high"
```

## 8. Run Tests

```bash
python -m unittest discover -s tests/adk_agent -p "test_*.py" -v
```

The current ADK-focused test suite covers 29 tests across:

- authentication
- audit logging
- monitoring persistence
- persistent runtime fallback
- Gemini retry/rate pacing
- tool wrappers
- FastAPI model-server route design
- Gemini-equivalent cost evidence

## 9. Evidence Files

| Evidence | Default location |
| --- | --- |
| Monitoring events, demo runs, model-cost events | `data/adk_monitoring.db` |
| ADK sessions when database sessions are available | `data/adk_sessions.db` |
| User audit log | `logs/adk_audit.log` |
| Runtime log | `logs/adk_agent.log` |
| Model metrics and training summaries | `artifacts/` |

## 10. Large Model Artefacts

Fine-tuned small-language-model artefacts may be stored outside the Git repository because model files can be large. The report appendix records the review archive location. If artefacts are unavailable locally, the API route tests still use fakes and do not require full model loading.

## 11. Troubleshooting

Model server unreachable:

- Start `python -m src.adk_agent.model_server`.
- Check `MODEL_SERVER_URL`.
- Confirm port `8100` is available.

Gemini 429 quota errors:

- Increase `GEMINI_MIN_CALL_INTERVAL_SECONDS`.
- Keep retry settings enabled.
- Avoid running multiple dashboard/server processes that call Gemini concurrently.

Dashboard login fails:

- Check `ADK_DASHBOARD_USERS` format: `username:<password>:role`.
- Valid roles are `admin`, `data_scientist`, `marketing_manager`, and `viewer`.

No monitoring rows appear:

- Run a live query or Alpha/Beta scenario first.
- Confirm `data/adk_monitoring.db` is writable.

Large models fail to load:

- Confirm artefact folders exist under `artifacts/`.
- Use tests or dashboard pages that rely on fake/model-server status when full local inference is not required.
