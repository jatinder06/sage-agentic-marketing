"""Content strategy recommendation tools using the fine-tuned LoRA-adapted LLM.

Model: Qwen2.5-0.5B-Instruct + LoRA adapter
Artifact: artifacts/content_llm_production/
Accuracy: 98.2% | Labels: DISCOUNT_REENGAGE, EDUCATIONAL_NURTURE, LOYALTY_UPSELL
"""

from typing import Dict, Any

from src.adk_agent.tools.model_client import predict


def recommend_content_strategy(customer_profile: str) -> Dict[str, Any]:
    """Recommend a marketing content strategy for a customer profile.

    Args:
        customer_profile: Text description of the customer profile including
            segment, engagement metrics, purchase history, and churn risk.

    Returns:
        Dictionary with recommended strategy
        (DISCOUNT_REENGAGE, EDUCATIONAL_NURTURE, or LOYALTY_UPSELL).
    """
    try:
        if not customer_profile or not customer_profile.strip():
            return {"status": "error", "error": "Customer profile is empty."}

        result = predict("/predict/content", {"text": customer_profile.strip()})

        return {
            "status": "success" if result["success"] else "error",
            "strategy": result.get("strategy"),
            "schema_valid": result.get("schema_valid", False),
            "model": "qwen2.5-content-lora",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
