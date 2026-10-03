"""Product recommendation tools using the fine-tuned LoRA-adapted LLM.

Model: Qwen2.5-0.5B-Instruct + LoRA adapter
Artifact: artifacts/recommendation_llm_production/
Accuracy: 74.0% | F1: 0.73 | Labels: CONSIDER, DO_NOT_RECOMMEND, RECOMMEND
"""

from typing import Dict, Any, List

from src.adk_agent.tools.model_client import predict


def recommend_product_action(product_review: str) -> Dict[str, Any]:
    """Determine a recommendation action based on a product review.

    Args:
        product_review: Customer product review text describing their experience.

    Returns:
        Dictionary with action (RECOMMEND, CONSIDER, or DO_NOT_RECOMMEND).
    """
    try:
        if not product_review or not product_review.strip():
            return {"status": "error", "error": "Product review text is empty."}

        result = predict("/predict/recommendation", {"text": product_review.strip()})

        return {
            "status": "success" if result["success"] else "error",
            "action": result.get("action"),
            "schema_valid": result.get("schema_valid", False),
            "model": "qwen2.5-recommendation-lora",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def recommend_product_actions_batch(product_reviews: List[str]) -> Dict[str, Any]:
    """Determine recommendation actions for multiple product reviews.

    Args:
        product_reviews: List of product review texts.

    Returns:
        Dictionary with per-review actions and aggregate distribution.
    """
    try:
        if not product_reviews:
            return {"status": "error", "error": "No product reviews provided."}

        clean = [r.strip() for r in product_reviews if r and r.strip()]
        if not clean:
            return {"status": "error", "error": "All review texts are empty."}

        results = predict("/predict/recommendation/batch", {"texts": clean})

        counts: Dict[str, int] = {}
        for r in results:
            action = r.get("action")
            if action:
                counts[action] = counts.get(action, 0) + 1

        success_count = sum(1 for r in results if r.get("success"))

        return {
            "status": "success" if success_count > 0 else "error",
            "total_processed": len(results),
            "successful": success_count,
            "distribution": counts,
            "results": [
                {"action": r.get("action"), "schema_valid": r.get("schema_valid", False)}
                for r in results[:20]
            ],
            "model": "qwen2.5-recommendation-lora",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
