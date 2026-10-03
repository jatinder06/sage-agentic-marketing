
# Recommendation LLM tools for the multi-agent system (production)

from pathlib import Path
import sys
from typing import Dict, Any

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experimentation.artifacts.recommendation_llm_production.final_model.recommendation_agent_llm import RecommendationLLMAgent

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = RecommendationLLMAgent(
            base_model_ref='/Users/jatinder.singh/drive/ai_mkt/models',
            adapter_path=str(project_root / "experimentation" / "artifacts" / 'recommendation_llm_production' / "final_model" / "lora_adapter"),
            labels=['CONSIDER', 'DO_NOT_RECOMMEND', 'RECOMMEND'],
            gguf_file='qwen2.5-0.5b-instruct-q5_0.gguf',
        )
    return _MODEL


def predict_recommendation_llm(input_text: str) -> Dict[str, Any]:
    try:
        model = _load_model()
        result = model.predict(input_text)
        return {"success": result["success"], **result}
    except Exception as e:
        return {"success": False, "error": str(e)}
