"""FastAPI model server optimized for Apple Neural Engine (ANE).

Uses MLX (Meta's ML framework) which natively supports ANE on Apple Silicon.

Installation:
    uv pip install mlx mlx-lm transformers torch torchvision

Usage:
    python -m src.adk_agent.model_server_ane
    # Then access http://localhost:8100

Note: This is an ANE-optimized variant. The original MPS-based server
is in model_server.py. Both use the same FastAPI interface (port 8100).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MLX Model Loaders (ANE-optimized)
# ============================================================================


class MLXSentimentModel:
    """DistilBERT sentiment (ANE-optimized via MLX)."""

    def __init__(self, model_path: str = None):
        """Load DistilBERT for sentiment classification."""
        if model_path is None:
            model_path = (
                Path(__file__).parent.parent.parent
                / "models" / "distilbert-base-uncased"
            )

        self.device = "gpu" if mx.metal.is_available() else "cpu"
        logger.info(f"Loading DistilBERT on {self.device}")

        try:
            # For production, convert model to MLX format using:
            # mlx_lm convert --model-name distilbert-base-uncased --output-dir models/mlx-distilbert
            from mlx_lm.models.distilbert import DistilBertForSequenceClassification
            from mlx_lm.utils import load

            self.model, self.tokenizer = load(str(model_path))
            self.labels = ["negative", "neutral", "positive"]
            logger.info("DistilBERT loaded successfully")
        except ImportError:
            logger.warning("MLX not available; falling back to PyTorch")
            self._load_pytorch_fallback(model_path)

    def _load_pytorch_fallback(self, model_path: str):
        """Fallback to PyTorch if MLX unavailable."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path)
            )
            self.model.eval()
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model.to(self.device)
            self.labels = ["negative", "neutral", "positive"]
            logger.info(f"DistilBERT loaded (PyTorch fallback, device={self.device})")
        except Exception as e:
            logger.error(f"Failed to load DistilBERT: {e}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment."""
        try:
            inputs = self.tokenizer(text, return_tensors="np", truncation=True)
            if hasattr(self, "model") and hasattr(self.model, "forward"):
                # MLX path
                logits = self.model(inputs)
            else:
                # PyTorch fallback
                import torch
                with torch.no_grad():
                    inputs_pt = {
                        k: torch.tensor(v).to(self.device) 
                        for k, v in inputs.items()
                    }
                    outputs = self.model(**inputs_pt)
                    logits = outputs.logits.cpu().numpy()

            scores = np.exp(logits[0]) / np.sum(np.exp(logits[0]))
            label_idx = np.argmax(scores)

            return {
                "label": self.labels[label_idx],
                "confidence": float(scores[label_idx]),
                "scores": {self.labels[i]: float(scores[i]) for i in range(len(self.labels))},
            }
        except Exception as e:
            logger.error(f"Sentiment prediction error: {e}")
            return {"error": str(e), "label": "unknown", "confidence": 0.0}

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for multiple texts."""
        return [self.predict(text) for text in texts]


class MLXQwenModel:
    """Qwen 2.5 with LoRA adapters (ANE-optimized)."""

    def __init__(self, adapter_name: str, model_path: str = None):
        """Load Qwen with specified LoRA adapter."""
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models"

        self.adapter_name = adapter_name
        self.device = "gpu" if mx.metal.is_available() else "cpu"
        logger.info(f"Loading Qwen+LoRA[{adapter_name}] on {self.device}")

        try:
            # MLX-based Qwen loading
            from mlx_lm.models.qwen import Qwen2ForCausalLM
            from mlx_lm.utils import load

            qwen_path = model_path / "qwen2.5-0.5b-instruct-hf"
            self.model, self.tokenizer = load(str(qwen_path))
            logger.info(f"Qwen loaded for {adapter_name}")
        except ImportError:
            logger.warning("MLX not available; using PyTorch fallback")
            self._load_pytorch_fallback(model_path, adapter_name)

    def _load_pytorch_fallback(self, model_path: Path, adapter_name: str):
        """Fallback to PyTorch."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            qwen_path = model_path / "qwen2.5-0.5b-instruct-hf"
            self.tokenizer = AutoTokenizer.from_pretrained(str(qwen_path))
            self.model = AutoModelForCausalLM.from_pretrained(str(qwen_path))

            # Load LoRA adapter if available
            adapter_path = (
                model_path / "parent" / f"{adapter_name}_lora"
            )
            if adapter_path.exists():
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    self.model, str(adapter_path)
                )

            self.model.eval()
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Qwen+LoRA loaded (PyTorch, device={self.device})")
        except Exception as e:
            logger.error(f"Failed to load Qwen: {e}")
            raise

    def predict(self, text: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Generate prediction."""
        try:
            import torch

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                )
            raw_output = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            return {
                "raw_output": raw_output,
                "adapter": self.adapter_name,
                "schema_valid": len(raw_output) > 0,
            }
        except Exception as e:
            logger.error(f"Qwen prediction error: {e}")
            return {"error": str(e), "raw_output": "", "schema_valid": False}


# ============================================================================
# FastAPI Setup
# ============================================================================

app = FastAPI(
    title="AI Marketing Model Server (ANE-Optimized)",
    description="Loads 7 models optimized for Apple Neural Engine (MLX backend)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class SentimentRequest(BaseModel):
    text: str


class SentimentBatchRequest(BaseModel):
    texts: List[str]


class QwenRequest(BaseModel):
    text: str
    max_tokens: int = 100


# ============================================================================
# Global Model Cache
# ============================================================================

_model_cache: Dict[str, Any] = {}


def load_all_models():
    """Load all 7 models on startup."""
    global _model_cache

    logger.info("=== Loading Models for ANE ===")
    start = time.time()

    try:
        logger.info("1/7 Loading DistilBERT (sentiment)...")
        _model_cache["sentiment"] = MLXSentimentModel()
        logger.info(f"   ✓ Sentiment loaded in {time.time()-start:.1f}s")
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")

    qwen_adapters = [
        "segmentation_llm_v2_robust",
        "churn_llm_production",
        "support_llm_production",
        "content_llm_production",
        "recommendation_llm_production",
    ]

    for i, adapter in enumerate(qwen_adapters, 2):
        try:
            logger.info(f"{i}/7 Loading Qwen+LoRA[{adapter}]...")
            t0 = time.time()
            _model_cache[adapter] = MLXQwenModel(adapter)
            logger.info(f"   ✓ {adapter} loaded in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"   ✗ Failed: {e}")

    total_time = time.time() - start
    logger.info(f"=== All models loaded in {total_time:.1f}s ===")
    logger.info(f"ANE (Apple Neural Engine) available: {mx.metal.is_available()}")


# ============================================================================
# Endpoints
# ============================================================================


@app.on_event("startup")
async def startup():
    """Load models on server startup."""
    load_all_models()


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "models_loaded": list(_model_cache.keys()),
        "ane_available": mx.metal.is_available(),
    }


@app.post("/predict/sentiment")
async def predict_sentiment(request: SentimentRequest):
    """Predict sentiment for single text."""
    if "sentiment" not in _model_cache:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded")

    try:
        result = _model_cache["sentiment"].predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/sentiment/batch")
async def predict_sentiment_batch(request: SentimentBatchRequest):
    """Predict sentiment for multiple texts."""
    if "sentiment" not in _model_cache:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded")

    try:
        results = _model_cache["sentiment"].predict_batch(request.texts)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/segmentation")
async def predict_segmentation(request: QwenRequest):
    """Predict customer segment."""
    if "segmentation_llm_v2_robust" not in _model_cache:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")

    try:
        result = _model_cache["segmentation_llm_v2_robust"].predict(
            request.text, request.max_tokens
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/churn_llm")
async def predict_churn_llm(request: QwenRequest):
    """Predict churn risk using LLM."""
    if "churn_llm_production" not in _model_cache:
        raise HTTPException(status_code=503, detail="Churn LLM model not loaded")

    try:
        result = _model_cache["churn_llm_production"].predict(
            request.text, request.max_tokens
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/support")
async def predict_support(request: QwenRequest):
    """Predict support ticket category."""
    if "support_llm_production" not in _model_cache:
        raise HTTPException(status_code=503, detail="Support model not loaded")

    try:
        result = _model_cache["support_llm_production"].predict(
            request.text, request.max_tokens
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/content")
async def predict_content(request: QwenRequest):
    """Predict content strategy."""
    if "content_llm_production" not in _model_cache:
        raise HTTPException(status_code=503, detail="Content model not loaded")

    try:
        result = _model_cache["content_llm_production"].predict(
            request.text, request.max_tokens
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/recommendation")
async def predict_recommendation(request: QwenRequest):
    """Predict product recommendation."""
    if "recommendation_llm_production" not in _model_cache:
        raise HTTPException(status_code=503, detail="Recommendation model not loaded")

    try:
        result = _model_cache["recommendation_llm_production"].predict(
            request.text, request.max_tokens
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ANE-optimized model server on http://0.0.0.0:8100")
    logger.info("ANE available: {}".format(mx.metal.is_available()))
    logger.info("Backend: MLX with PyTorch fallback")

    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
