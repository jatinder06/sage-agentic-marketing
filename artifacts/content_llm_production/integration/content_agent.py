
# Content strategy agent that uses the fine-tuned LLM tool

from typing import Any, Dict
from src.agents.tools.content_llm_tools import predict_content_strategy_llm


class ContentAgent:
    name = "content_agent"

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        input_text = payload["input_text"]
        result = predict_content_strategy_llm(input_text)
        return {"agent": self.name, **result}
