
# Churn risk agent that uses the fine-tuned LLM tool

from typing import Any, Dict
from src.agents.tools.churn_llm_tools import predict_churn_risk_llm


class ChurnAgent:
    name = "churn_agent"

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        input_text = payload["input_text"]
        result = predict_churn_risk_llm(input_text)
        return {"agent": self.name, **result}
