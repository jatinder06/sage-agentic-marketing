"""Customer segmentation tools using the fine-tuned LoRA-adapted LLM (v2 robust).

Model: Qwen2.5-0.5B-Instruct + LoRA adapter
Artifact: artifacts/segmentation_llm_v2_robust/
Accuracy: 91.2% | Labels: At Risk, Champions, Loyal, Needs Attention
"""

from typing import Dict, Any, List

from src.adk_agent.tools.model_client import predict


def segment_customer(customer_profile: str) -> Dict[str, Any]:
    """Classify a customer into a marketing segment based on their RFM profile.

    Args:
        customer_profile: Serialized customer RFM data, e.g.
            'recency=15 days; frequency=8 orders; monetary=1250.00;
             avg_order_value=156.25; tenure=24 months'

    Returns:
        Dictionary with segment label (At Risk, Champions, Loyal, Needs Attention).
    """
    try:
        if not customer_profile or not customer_profile.strip():
            return {"status": "error", "error": "Customer profile text is empty."}

        result = predict("/predict/segmentation", {"text": customer_profile.strip()})

        return {
            "status": "success" if result["success"] else "error",
            "segment": result.get("segment"),
            "schema_valid": result.get("schema_valid", False),
            "model": "qwen2.5-segmentation-lora-v2",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def segment_customers_batch(customer_profiles: List[str]) -> Dict[str, Any]:
    """Classify multiple customers into marketing segments.

    Args:
        customer_profiles: List of serialized customer RFM profile strings.

    Returns:
        Dictionary with per-customer segment results and distribution.
    """
    try:
        if not customer_profiles:
            return {"status": "error", "error": "No customer profiles provided."}

        clean = [p.strip() for p in customer_profiles if p and p.strip()]
        if not clean:
            return {"status": "error", "error": "All profile texts are empty."}

        results = predict("/predict/segmentation/batch", {"texts": clean})

        counts: Dict[str, int] = {}
        for r in results:
            seg = r.get("segment")
            if seg:
                counts[seg] = counts.get(seg, 0) + 1

        success_count = sum(1 for r in results if r.get("success"))

        return {
            "status": "success" if success_count > 0 else "error",
            "total_processed": len(results),
            "successful": success_count,
            "distribution": counts,
            "results": [
                {"segment": r.get("segment"), "schema_valid": r.get("schema_valid", False)}
                for r in results[:20]
            ],
            "model": "qwen2.5-segmentation-lora-v2",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
