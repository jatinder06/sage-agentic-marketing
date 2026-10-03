
# Segmentation agent that uses the fine-tuned LLM tool (v2 — robust)

from typing import Any, Dict
from src.agents.tools.segmentation_llm_tools import predict_customer_segment_llm


class SegmentationAgent:
    name = "segmentation_agent"

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        input_text = payload["input_text"]
        result = predict_customer_segment_llm(input_text)
        return {"agent": self.name, **result}
