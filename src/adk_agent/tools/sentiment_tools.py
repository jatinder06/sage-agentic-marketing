"""Sentiment analysis tools using the fine-tuned DistilBERT model.

Model: distilbert-base-uncased fine-tuned on Amazon reviews (3-class).
Artifact: artifacts/distilbert_sentiment_production/
Accuracy: 79.8% | ROC-AUC: 0.93
"""

from typing import Dict, Any, List

from src.adk_agent.tools.model_client import predict


def analyze_sentiment(review_text: str) -> Dict[str, Any]:
    """Analyze sentiment of a single customer review.

    Args:
        review_text: The review or feedback text to analyze.

    Returns:
        Dictionary with label (negative/neutral/positive), confidence score,
        and per-class probability scores.
    """
    try:
        if not review_text or not review_text.strip():
            return {"status": "error", "error": "Review text is empty."}

        result = predict("/predict/sentiment", {"text": review_text.strip()})

        return {
            "status": "success",
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "scores": {k: round(v, 4) for k, v in result["scores"].items()},
            "model": "distilbert-sentiment-finetuned",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def analyze_sentiment_batch(reviews: List[str]) -> Dict[str, Any]:
    """Analyze sentiment of multiple reviews in batch.

    Args:
        reviews: List of review texts to analyze.

    Returns:
        Dictionary with per-review results and aggregate distribution.
    """
    try:
        if not reviews:
            return {"status": "error", "error": "No reviews provided."}

        clean = [r.strip() for r in reviews if r and r.strip()]
        if not clean:
            return {"status": "error", "error": "All review texts are empty."}

        results = predict("/predict/sentiment/batch", {"texts": clean})

        counts = {"negative": 0, "neutral": 0, "positive": 0}
        for r in results:
            counts[r["label"]] = counts.get(r["label"], 0) + 1

        return {
            "status": "success",
            "total_analyzed": len(results),
            "distribution": counts,
            "average_confidence": round(
                sum(r["confidence"] for r in results) / len(results), 4
            ),
            "results": [
                {"label": r["label"], "confidence": round(r["confidence"], 4)}
                for r in results[:20]
            ],
            "model": "distilbert-sentiment-finetuned",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
