# System Documentation

This folder contains public-facing documentation for the ADK-based Digital Marketing Intelligence System implemented in `src/adk_agent`.

The system demonstrates customer-experience decision support through six specialist marketing capabilities:

- sentiment analysis
- churn prediction
- customer segmentation
- support intent routing
- content strategy recommendation
- product recommendation action

The current implementation is an academic/local system. It does not connect to live CRM, email, advertising, or social-media platforms.

## Documents

1. [HOWTO.md](./HOWTO.md)  
   Setup, run commands, environment variables, tests, and troubleshooting.

2. [USER_GUIDE.md](./USER_GUIDE.md)  
   Role-based guide for using the dashboard, live agent tester, and Alpha/Beta demo pages.

3. [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)  
   Architecture, module map, persistence design, tool wrappers, testing, and extension notes.

4. [SYSTEM_STORY_ALPHA_BETA.md](./SYSTEM_STORY_ALPHA_BETA.md)  
   Customer-experience narrative for the Alpha product-company and Beta service-company demonstrations.

5. [ESSENTIAL_MARKETING_NARRATIVE_GUIDE.md](./ESSENTIAL_MARKETING_NARRATIVE_GUIDE.md)  
   Short business-language guide for explaining the project in presentations.

## Current Implementation Paths

| Area | Path |
| --- | --- |
| ADK root agent and specialist agents | `src/adk_agent/agent.py` |
| Streamlit dashboard | `src/adk_agent/dashboard.py` |
| Standalone Streamlit demo | `src/adk_agent/demo.py` |
| FastAPI model server | `src/adk_agent/model_server.py` |
| Tool wrappers | `src/adk_agent/tools/` |
| Monitoring store | `src/adk_agent/monitoring_store.py` |
| Audit logging | `src/adk_agent/audit.py` |
| Persistent ADK runtime helper | `src/adk_agent/persistent_runtime.py` |
| Unit tests | `tests/adk_agent/` |
| Model training/evaluation artefacts | `artifacts/` |

## Quick Start

Run from the repository root.

```bash
# Terminal 1: start local model inference server
python -m src.adk_agent.model_server

# Terminal 2: start the main dashboard
streamlit run src/adk_agent/dashboard.py
```

Optional standalone demo:

```bash
streamlit run src/adk_agent/demo.py
```

Run tests:

```bash
python -m unittest discover -s tests/adk_agent -p "test_*.py" -v
```

## Public Repo Scope

This documentation intentionally describes the current ADK implementation only. Older documentation for previous architecture experiments has been removed because those paths and capabilities are not part of the current public release.
