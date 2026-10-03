
# Support intent agent that uses the fine-tuned LLM tool

from typing import Any, Dict
from src.agents.tools.support_llm_tools import predict_support_intent_llm


class SupportAgent:
    name = "support_agent"

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        input_text = payload["input_text"]
        result = predict_support_intent_llm(input_text)
        return {"agent": self.name, **result}
