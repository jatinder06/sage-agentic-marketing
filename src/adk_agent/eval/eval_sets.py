"""Evaluation test cases for the ADK multi-agent system.

Defines EvalSet with test cases covering each sub-agent's tool invocation
and the root orchestrator's routing behaviour.

ADK eval structure:
  EvalCase.conversation = list[Invocation]
  Invocation.user_content = Content (the user message)
  Invocation.intermediate_data = IntermediateData (expected tool calls)
  IntermediateData.tool_uses = list[FunctionCall]
"""

from __future__ import annotations

from google.adk.evaluation.eval_case import EvalCase, Invocation, IntermediateData
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types


def _invocation(user_text: str, tool_name: str | None = None) -> Invocation:
    """Helper to build an Invocation with optional expected tool call."""
    kwargs = {
        "invocation_id": "",
        "user_content": types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_text)],
        ),
    }
    if tool_name:
        kwargs["intermediate_data"] = IntermediateData(
            tool_uses=[types.FunctionCall(name=tool_name, args={})],
        )
    return Invocation(**kwargs)


# ---------------------------------------------------------------------------
# Sentiment Agent eval cases
# ---------------------------------------------------------------------------

sentiment_cases = [
    EvalCase(
        eval_id="sentiment_positive",
        conversation=[
            _invocation(
                "Analyze the sentiment of this review: "
                "'This product is absolutely amazing! Best purchase I ever made.'",
                tool_name="analyze_sentiment",
            )
        ],
    ),
    EvalCase(
        eval_id="sentiment_negative",
        conversation=[
            _invocation(
                "What is the sentiment of: "
                "'Terrible quality, broke after one day. Complete waste of money.'",
                tool_name="analyze_sentiment",
            )
        ],
    ),
    EvalCase(
        eval_id="sentiment_batch",
        conversation=[
            _invocation(
                "Analyze sentiment for these reviews: "
                "1) 'Great product, love it!' "
                "2) 'It was okay, nothing special.' "
                "3) 'Worst experience ever.'",
                tool_name="analyze_sentiment_batch",
            )
        ],
    ),
]

# ---------------------------------------------------------------------------
# Churn Agent eval cases
# ---------------------------------------------------------------------------

churn_cases = [
    EvalCase(
        eval_id="churn_xgboost_high_risk",
        conversation=[
            _invocation(
                "Predict churn for a customer with: tenure=2 months, "
                "monthly_charges=89.50, total_charges=179.00, "
                "contract=Month-to-month, internet_service=Fiber optic, "
                "payment_method=Electronic check, no online security or tech support.",
                tool_name="predict_churn_xgboost",
            )
        ],
    ),
    EvalCase(
        eval_id="churn_llm_text",
        conversation=[
            _invocation(
                "Is this customer at risk of churning? "
                "tenure=59 months; contract=One year; internet=Fiber optic; "
                "monthly_charges=85.5; online_security=Yes; tech_support=No",
                tool_name="predict_churn_llm",
            )
        ],
    ),
]

# ---------------------------------------------------------------------------
# Segmentation Agent eval cases
# ---------------------------------------------------------------------------

segmentation_cases = [
    EvalCase(
        eval_id="segmentation_champion",
        conversation=[
            _invocation(
                "Segment this customer: recency=5 days, frequency=45 orders, "
                "monetary=$12,500, average order value=$278, last purchase=last week.",
                tool_name="segment_customer",
            )
        ],
    ),
    EvalCase(
        eval_id="segmentation_at_risk",
        conversation=[
            _invocation(
                "What segment does this customer belong to? "
                "recency=120 days, frequency=3 orders, monetary=$150, "
                "last purchase was 4 months ago, declining engagement.",
                tool_name="segment_customer",
            )
        ],
    ),
]

# ---------------------------------------------------------------------------
# Support Agent eval cases
# ---------------------------------------------------------------------------

support_cases = [
    EvalCase(
        eval_id="support_billing",
        conversation=[
            _invocation(
                "Classify this support query: "
                "'I was charged twice for my subscription this month. "
                "Please refund the duplicate charge.'",
                tool_name="classify_support_intent",
            )
        ],
    ),
    EvalCase(
        eval_id="support_technical",
        conversation=[
            _invocation(
                "Route this ticket: "
                "'The app keeps crashing when I try to upload files larger than 10MB. "
                "I get error code E-5021.'",
                tool_name="classify_support_intent",
            )
        ],
    ),
]

# ---------------------------------------------------------------------------
# Content Strategy Agent eval cases
# ---------------------------------------------------------------------------

content_cases = [
    EvalCase(
        eval_id="content_discount_reengage",
        conversation=[
            _invocation(
                "What content strategy should we use for this customer? "
                "Segment: At Risk, last purchase 90 days ago, declining engagement, "
                "previously high spender, unsubscribed from emails.",
                tool_name="recommend_content_strategy",
            )
        ],
    ),
    EvalCase(
        eval_id="content_loyalty_upsell",
        conversation=[
            _invocation(
                "Recommend a marketing strategy for: "
                "Segment: Champions, monthly spend=$500, loyalty member, "
                "high engagement, 30+ purchases in the last year.",
                tool_name="recommend_content_strategy",
            )
        ],
    ),
]

# ---------------------------------------------------------------------------
# Recommendation Agent eval cases
# ---------------------------------------------------------------------------

recommendation_cases = [
    EvalCase(
        eval_id="recommendation_positive",
        conversation=[
            _invocation(
                "Should we recommend this product based on the review: "
                "'Exceeded all my expectations. Premium build quality, fast delivery, "
                "and excellent customer service. Five stars!'",
                tool_name="recommend_product_action",
            )
        ],
    ),
    EvalCase(
        eval_id="recommendation_negative",
        conversation=[
            _invocation(
                "Based on this review, should we promote this product? "
                "'Do not buy this. Arrived damaged, customer service was unhelpful, "
                "and the product doesn't match the description at all.'",
                tool_name="recommend_product_action",
            )
        ],
    ),
]

# ---------------------------------------------------------------------------
# Multi-agent routing eval cases (no specific tool expected, just routing)
# ---------------------------------------------------------------------------

routing_cases = [
    EvalCase(
        eval_id="routing_sentiment",
        conversation=[
            _invocation("What do customers feel about our new laptop?"),
        ],
    ),
    EvalCase(
        eval_id="routing_churn",
        conversation=[
            _invocation(
                "Will this customer leave us? They have a month-to-month contract "
                "and have only been with us 3 months.",
            ),
        ],
    ),
    EvalCase(
        eval_id="routing_support",
        conversation=[
            _invocation(
                "I need to classify this ticket: 'My account was locked after "
                "too many password attempts.'",
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Complete eval set
# ---------------------------------------------------------------------------

def get_full_eval_set() -> EvalSet:
    """Return the complete evaluation set covering all agents."""
    all_cases = (
        sentiment_cases
        + churn_cases
        + segmentation_cases
        + support_cases
        + content_cases
        + recommendation_cases
        + routing_cases
    )
    return EvalSet(
        eval_set_id="ai_marketing_full",
        name="AI Marketing Multi-Agent Evaluation",
        description=(
            "End-to-end evaluation of all 6 sub-agents and orchestrator routing "
            "for the AI Marketing ADK system."
        ),
        eval_cases=all_cases,
    )


def get_agent_eval_set(agent_name: str) -> EvalSet:
    """Return an evaluation set for a specific agent."""
    case_map = {
        "sentiment": sentiment_cases,
        "churn": churn_cases,
        "segmentation": segmentation_cases,
        "support": support_cases,
        "content": content_cases,
        "recommendation": recommendation_cases,
        "routing": routing_cases,
    }
    cases = case_map.get(agent_name, [])
    if not cases:
        raise ValueError(
            f"Unknown agent: {agent_name}. "
            f"Available: {', '.join(case_map.keys())}"
        )
    return EvalSet(
        eval_set_id=f"ai_marketing_{agent_name}",
        name=f"AI Marketing - {agent_name.title()} Agent Evaluation",
        eval_cases=cases,
    )
