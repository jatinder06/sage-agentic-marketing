"""AI Marketing Multi-Agent System — Google ADK.

Orchestrates six specialised sub-agents, each backed by a fine-tuned model
from artifacts/:

  1. Sentiment Agent      — DistilBERT  (3-class: negative/neutral/positive)
  2. Churn Agent          — XGBoost + Qwen-LoRA (binary: HIGH_RISK/LOW_RISK)
  3. Segmentation Agent   — Qwen-LoRA v2 (4-class: At Risk/Champions/Loyal/Needs Attention)
  4. Support Agent        — Qwen-LoRA (5-class: ACCOUNT/BILLING/DELIVERY/GENERAL/TECHNICAL)
  5. Content Agent        — Qwen-LoRA (3-class: DISCOUNT_REENGAGE/EDUCATIONAL_NURTURE/LOYALTY_UPSELL)
  6. Recommendation Agent — Qwen-LoRA (3-class: RECOMMEND/CONSIDER/DO_NOT_RECOMMEND)

Run:
    adk web src.adk_agent
    adk run src.adk_agent
"""

import logging
import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from google.adk.agents import LlmAgent

from src.adk_agent.gemini_retry import install_gemini_retry_patch
from src.adk_agent.logging_config import configure_adk_logging


class _SuppressGenAIPartWarning(logging.Filter):
    """Hide noisy google-genai warning for function-call parts."""

    _PREFIX = "Warning: there are non-text parts in the response:"

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not (
            record.name == "google_genai.types"
            and isinstance(msg, str)
            and msg.startswith(self._PREFIX)
        )


_genai_logger = logging.getLogger("google_genai.types")
if not any(isinstance(f, _SuppressGenAIPartWarning) for f in _genai_logger.filters):
    _genai_logger.addFilter(_SuppressGenAIPartWarning())

load_dotenv()
configure_adk_logging()

install_gemini_retry_patch(
    retries=int(os.getenv("GEMINI_RETRY_ATTEMPTS", "5")),
    backoff_factor=int(os.getenv("GEMINI_RETRY_BACKOFF_FACTOR", "2")),
    min_interval_seconds=float(os.getenv("GEMINI_MIN_CALL_INTERVAL_SECONDS", "4")),
)


def _load_config() -> Dict[str, Any]:
    config_path = os.getenv("ADK_CONFIG_PATH")
    if not config_path:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(repo_root, "configs", "config.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


_CONFIG = _load_config()


def _get_config_value(path: tuple[str, ...], default: Any = None) -> Any:
    node: Any = _CONFIG
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node

USE_LITELLM = os.getenv("USE_LITELLM", "false").lower() == "true"

DEFAULT_GEMINI_MODEL = _get_config_value(
    ("adk_agent", "default_model_gemini"), "gemini-2.0-flash"
)

SUB_AGENT_MODEL_GEMINI = os.getenv(
    "SUB_AGENT_MODEL_GEMINI",
    _get_config_value(("adk_agent", "sub_agent_model_gemini"), DEFAULT_GEMINI_MODEL),
)

if USE_LITELLM:
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.models.registry import LLMRegistry

    LLMRegistry._register(r"openrouter/.*", LiteLlm)
    ORCHESTRATOR_MODEL = os.getenv(
        "ORCHESTRATOR_MODEL_LITELLM",
        _get_config_value(
            ("adk_agent", "orchestrator_model_litellm"),
            "openrouter/google/gemma-4-31b-it:free",
        ),
    )
else:
    ORCHESTRATOR_MODEL = os.getenv(
        "ORCHESTRATOR_MODEL_GEMINI",
        _get_config_value(("adk_agent", "orchestrator_model_gemini"), DEFAULT_GEMINI_MODEL),
    )

from src.adk_agent.callbacks import (
    before_model_callback,
    after_model_callback,
    on_model_error_callback,
    before_tool_callback,
    after_tool_callback,
    on_tool_error_callback,
)
from src.adk_agent.tools.sentiment_tools import (
    analyze_sentiment,
    analyze_sentiment_batch,
)
from src.adk_agent.tools.churn_tools import (
    predict_churn_xgboost,
    predict_churn_llm,
)
from src.adk_agent.tools.segmentation_tools import (
    segment_customer,
    segment_customers_batch,
)
from src.adk_agent.tools.support_tools import classify_support_intent
from src.adk_agent.tools.content_tools import recommend_content_strategy
from src.adk_agent.tools.recommendation_tools import (
    recommend_product_action,
    recommend_product_actions_batch,
)


# ---------------------------------------------------------------------------
# Sub-agent 1: Sentiment Analysis
# ---------------------------------------------------------------------------
sentiment_agent = LlmAgent(
    name="sentiment_agent",
    description=(
        "Analyzes customer review sentiment using a fine-tuned DistilBERT model. "
        "Use this agent when the user wants to understand customer feelings "
        "from reviews, feedback, or social media text."
    ),
    model=SUB_AGENT_MODEL_GEMINI,
    instruction="""You are the Sentiment Analysis Agent. You use a fine-tuned
DistilBERT model (79.8% accuracy, 0.93 ROC-AUC) trained on Amazon product
reviews to classify text into negative, neutral, or positive sentiment.

When the user provides review text:
1. Use analyze_sentiment for a single review.
2. Use analyze_sentiment_batch for multiple reviews.

Always report the sentiment label and confidence scores. Provide actionable
insights based on sentiment patterns.""",
    tools=[analyze_sentiment, analyze_sentiment_batch],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    on_model_error_callback=on_model_error_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_tool_error_callback=on_tool_error_callback,
)

# ---------------------------------------------------------------------------
# Sub-agent 2: Churn Prediction
# ---------------------------------------------------------------------------
churn_agent = LlmAgent(
    name="churn_agent",
    description=(
        "Predicts customer churn risk using an XGBoost model (structured data) "
        "or a fine-tuned LLM (text profiles). Use this agent when the user "
        "asks about customer retention, churn probability, or at-risk customers."
    ),
    model=SUB_AGENT_MODEL_GEMINI,
    instruction="""You are the Churn Prediction Agent. You have two models:

1. **XGBoost** (predict_churn_xgboost) — Takes structured customer
   features and returns a churn probability.

2. **LLM** (predict_churn_llm) — Takes a natural-language customer profile
   description and predicts HIGH_RISK or LOW_RISK.

Choose the right model based on what the user provides.
Always explain the risk assessment and suggest retention actions.""",
    tools=[predict_churn_xgboost, predict_churn_llm],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    on_model_error_callback=on_model_error_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_tool_error_callback=on_tool_error_callback,
)

# ---------------------------------------------------------------------------
# Sub-agent 3: Customer Segmentation
# ---------------------------------------------------------------------------
segmentation_agent = LlmAgent(
    name="segmentation_agent",
    description=(
        "Segments customers into marketing groups (Champions, Loyal, "
        "Needs Attention, At Risk) using a fine-tuned LLM. Use this agent "
        "when the user wants to classify customers by value or engagement."
    ),
    model=SUB_AGENT_MODEL_GEMINI,
    instruction="""You are the Customer Segmentation Agent. You classify
customers into four marketing segments:

- **Champions** — High value, reward loyalty.
- **Loyal** — Consistent, upsell.
- **Needs Attention** — Declining, re-engage.
- **At Risk** — Low activity, win-back.

Use segment_customer for individual customers or segment_customers_batch
for multiple profiles.

Always explain the RFM metrics (Recency, Frequency, Monetary) and segment
characteristics.""",
    tools=[segment_customer, segment_customers_batch],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    on_model_error_callback=on_model_error_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_tool_error_callback=on_tool_error_callback,
)

# ---------------------------------------------------------------------------
# Sub-agent 4: Support Intent Classification
# ---------------------------------------------------------------------------
support_agent = LlmAgent(
    name="support_agent",
    description=(
        "Classifies customer support queries into intent categories "
        "(ACCOUNT, BILLING, DELIVERY, GENERAL, TECHNICAL). Use this agent "
        "when the user wants to route or triage support tickets."
    ),
    model=SUB_AGENT_MODEL_GEMINI,
    instruction="""You are the Support Intent Agent. You classify customer
queries into five categories:

- **ACCOUNT** — Account access, profile, login.
- **BILLING** — Payment, charges, refunds.
- **DELIVERY** — Shipping, tracking, returns.
- **GENERAL** — Inquiries, feedback.
- **TECHNICAL** — Technical problems, bugs.

Use classify_support_intent with the query text.

Always provide the detected intent and recommended resolution steps.""",
    tools=[classify_support_intent],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    on_model_error_callback=on_model_error_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_tool_error_callback=on_tool_error_callback,
)

# ---------------------------------------------------------------------------
# Sub-agent 5: Content Strategy
# ---------------------------------------------------------------------------
content_agent = LlmAgent(
    name="content_agent",
    description=(
        "Recommends marketing content strategy for customer profiles "
        "(DISCOUNT_REENGAGE, EDUCATIONAL_NURTURE, LOYALTY_UPSELL). "
        "Use this agent for campaign planning and personalised marketing."
    ),
    model=SUB_AGENT_MODEL_GEMINI,
    instruction="""You are the Content Strategy Agent. You use a fine-tuned
Qwen2.5 LLM with LoRA adaptation (98.2% accuracy) to recommend one of
three marketing strategies:

- **DISCOUNT_REENGAGE** — Offer discounts/promotions to re-engage lapsed
  or at-risk customers. Target: low engagement, declining purchases.
- **EDUCATIONAL_NURTURE** — Send educational content, tips, tutorials to
  nurture newer or uncertain customers. Target: new/exploring customers.
- **LOYALTY_UPSELL** — Offer premium upgrades, exclusive deals to reward
  loyal high-value customers. Target: champions and loyal segments.

Use recommend_content_strategy with a customer profile description.
After recommending a strategy, suggest specific campaign ideas.""",
    tools=[recommend_content_strategy],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    on_model_error_callback=on_model_error_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_tool_error_callback=on_tool_error_callback,
)

# ---------------------------------------------------------------------------
# Sub-agent 6: Product Recommendation
# ---------------------------------------------------------------------------
recommendation_agent = LlmAgent(
    name="recommendation_agent",
    description=(
        "Determines product recommendation actions (RECOMMEND, CONSIDER, "
        "DO_NOT_RECOMMEND) from customer reviews. Use this agent when the "
        "user wants to decide whether to recommend a product based on review text."
    ),
    model=SUB_AGENT_MODEL_GEMINI,
    instruction="""You are the Product Recommendation Agent. You use a fine-tuned
Qwen2.5 LLM with LoRA adaptation (74.0% accuracy, F1: 0.73) to determine
a recommendation action from product review text:

- **RECOMMEND** — The review is strongly positive. The product should be
  recommended to other customers or promoted in marketing materials.
- **CONSIDER** — The review is mixed or neutral. The product may suit some
  customers but has notable caveats. Suggest it with qualifications.
- **DO_NOT_RECOMMEND** — The review is negative. The product should not be
  promoted and may need quality or service improvements.

Use recommend_product_action for a single review or
recommend_product_actions_batch for multiple reviews.

After classification, provide business context: for RECOMMEND products suggest
cross-sell opportunities; for DO_NOT_RECOMMEND flag quality issues to the
product team; for CONSIDER suggest targeted audiences who may still benefit.""",
    tools=[recommend_product_action, recommend_product_actions_batch],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    on_model_error_callback=on_model_error_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_tool_error_callback=on_tool_error_callback,
)

# ---------------------------------------------------------------------------
# Root Orchestrator Agent
# ---------------------------------------------------------------------------
root_agent = LlmAgent(
    name="ai_marketing_orchestrator",
    description="AI Marketing Orchestrator — routes queries to specialised agents.",
    model=ORCHESTRATOR_MODEL,
    instruction="""You are the AI Marketing Orchestrator, the central coordinator
of an AI-powered digital marketing system built for an MSc Data Science project.

You manage six specialised agents, each backed by a fine-tuned ML model:

1. **sentiment_agent** — Sentiment analysis on reviews/feedback (DistilBERT, 79.8% accuracy)
2. **churn_agent** — Customer churn risk prediction (XGBoost + LLM, up to 84.6% ROC-AUC)
3. **segmentation_agent** — Customer segmentation into 4 groups (LLM, 91.2% accuracy)
4. **support_agent** — Support ticket intent classification (LLM, 94.7% accuracy)
5. **content_agent** — Marketing content strategy recommendation (LLM, 98.2% accuracy)
6. **recommendation_agent** — Product recommendation from reviews (LLM, 74.0% accuracy)

**Routing rules:**
- Review/feedback sentiment → sentiment_agent
- Churn risk or retention questions → churn_agent
- Customer grouping/segmentation → segmentation_agent
- Support ticket routing → support_agent
- Campaign/content strategy → content_agent
- Product recommendation decisions from reviews → recommendation_agent

When delegating, always call the `transfer_to_agent` tool with the target
agent name. Do not call agent names as tools directly.

**Multi-step workflows:** For complex requests you can chain agents. For example:
- "Analyze this customer and recommend a strategy" →
    use `transfer_to_agent` to `segmentation_agent`, then `content_agent`
- "Is this customer at risk? What should we do?" → Use churn_agent for risk assessment, content_agent for strategy
- "Should we promote this product based on reviews?" → Use recommendation_agent, optionally followed by sentiment_agent

Always synthesize insights from multiple agents to provide comprehensive business recommendations.""",
    sub_agents=[
        sentiment_agent,
        churn_agent,
        segmentation_agent,
        support_agent,
        content_agent,
        recommendation_agent,
    ],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    on_model_error_callback=on_model_error_callback,
)
