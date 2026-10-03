"""Support intent classification tools using the fine-tuned LoRA-adapted LLM.

Model: Qwen2.5-0.5B-Instruct + LoRA adapter
Artifact: artifacts/support_llm_production/
Accuracy: 94.7% | Labels: ACCOUNT, BILLING, DELIVERY, GENERAL, TECHNICAL
"""

from typing import Dict, Any

from src.adk_agent.tools.model_client import predict


def classify_support_intent(customer_query: str) -> Dict[str, Any]:
    """Classify a customer support query into an intent category.

    Args:
        customer_query: The customer's support message or ticket text.

    Returns:
        Dictionary with intent tag (ACCOUNT, BILLING, DELIVERY, GENERAL, TECHNICAL).
    """
    try:
        if not customer_query or not customer_query.strip():
            return {"status": "error", "error": "Customer query is empty."}

        result = predict("/predict/support", {"text": customer_query.strip()})

        return {
            "status": "success" if result["success"] else "error",
            "intent": result.get("intent"),
            "schema_valid": result.get("schema_valid", False),
            "model": "qwen2.5-support-lora",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
