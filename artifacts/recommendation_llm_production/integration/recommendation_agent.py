
# Recommendation agent that uses the fine-tuned LLM tool (production)

from typing import Any, Dict
from src.agents.tools.recommendation_llm_tools import predict_recommendation_llm


class RecommendationAgent:
    name = "recommendation_agent"

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        input_text = payload["input_text"]
        result = predict_recommendation_llm(input_text)
        return {{"agent": self.name, **result}}
