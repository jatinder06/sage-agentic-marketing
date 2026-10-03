"""Lazy model loader with singleton caching for all fine-tuned experiment models.

All models are loaded from artifacts/ on first use and cached
for the lifetime of the process.  The five LoRA models share a single Qwen2.5
base — each adapter is merged on top at load time.  The GGUF base is loaded
only once to avoid a segfault in the GGUF de-quantizer on repeated loads.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

_cache: Dict[str, Any] = {}
_lock = threading.Lock()


def _resolve_torch_device() -> str | None:
    """Pick a stable Torch inference device, with env override support."""
    requested = os.getenv("ADK_TORCH_DEVICE") or os.getenv("TORCH_DEVICE")
    if requested:
        device = requested.strip().lower()
        if device:
            logger.info("Using Torch device override: %s", device)
            return device

    if sys.platform == "darwin":
        import torch
        if torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon GPU) for inference")
            return "mps"
        logger.info("MPS not available, falling back to CPU")
        return "cpu"

    return None


def _read_summary(artifact_dir: Path) -> Dict[str, Any]:
    for name in ("experiment_summary.json", "run_summary.json"):
        p = artifact_dir / name
        if p.exists():
            return json.loads(p.read_text())
    return {}


def _load_wrapper_from_file(wrapper_path: Path, class_name: str, **kwargs: Any) -> Any:
    if not wrapper_path.exists():
        raise FileNotFoundError(f"Model wrapper not found: {wrapper_path}")
    spec = importlib.util.spec_from_file_location(wrapper_path.stem, str(wrapper_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec from {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)
    return cls(**kwargs)


# ------------------------------------------------------------------
# Shared GGUF base model cache
# ------------------------------------------------------------------

_HF_CONVERTED_DIR = PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct-hf"


def _get_shared_tokenizer():
    """Load the tokenizer once and cache it."""
    key = "_shared_tokenizer"
    if key in _cache:
        return _cache[key]

    from transformers import AutoTokenizer

    if not _HF_CONVERTED_DIR.exists():
        raise FileNotFoundError(
            f"Pre-converted HF base model not found at {_HF_CONVERTED_DIR}. "
            f"Run the GGUF-to-HF conversion script first (see model_loader.py docstring)."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(_HF_CONVERTED_DIR), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _cache[key] = tokenizer
    return tokenizer


def _load_fresh_base():
    """Load a fresh base model from the pre-converted HF weights."""
    import torch
    from transformers import AutoModelForCausalLM

    logger.info("  Loading base model from %s …", _HF_CONVERTED_DIR.name)
    model = AutoModelForCausalLM.from_pretrained(
        str(_HF_CONVERTED_DIR), dtype=torch.float32, local_files_only=True
    )
    logger.info("  Base model loaded")
    return model


class _LoraInferenceWrapper:
    """Lightweight wrapper: shared base + per-task LoRA adapter."""

    def __init__(self, adapter_path: str, labels: list, device: str,
                 max_length: int = 384, max_new_tokens: int = 40,
                 prompt_template: str = "", parse_fn: Any = None):
        import torch
        from peft import PeftModel

        self.labels = labels
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = torch.device(device)
        self.tokenizer = _get_shared_tokenizer()
        self._prompt_template = prompt_template
        self._parse_fn = parse_fn

        base_model = _load_fresh_base()
        logger.info("  Merging LoRA adapter from %s …", Path(adapter_path).name)
        model = PeftModel.from_pretrained(
            base_model, adapter_path, local_files_only=True
        )
        self.model = model.to(self.device)
        self.model.eval()
        logger.info("  LoRA adapter merged, model ready on %s", self.device)

    def _format_prompt(self, input_text: str) -> str:
        return (
            "### Instruction:\n"
            f"{self._prompt_template}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )

    def _default_parse(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}

    def predict(self, input_text: str) -> Dict[str, Any]:
        import torch

        prompt = self._format_prompt(input_text)
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_ids = out[0][enc["input_ids"].shape[1]:]
        generated = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        if self._parse_fn:
            parsed = self._parse_fn(generated, self.labels)
        else:
            parsed = self._default_parse(generated)

        return {
            "success": True,
            "raw_output": generated,
            "schema_valid": bool(parsed),
            **parsed,
        }

    def predict_batch(self, inputs: list) -> list:
        return [self.predict(x) for x in inputs]


# ------------------------------------------------------------------
# Parse helpers per model type
# ------------------------------------------------------------------

def _parse_churn(text: str, labels: list) -> dict:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            risk = str(obj.get("risk", "")).strip()
            if risk in labels:
                return {"risk": risk, "schema_valid": True}
        except Exception:
            pass
    upper = text.upper()
    for lbl in labels:
        if lbl.upper() in upper:
            return {"risk": lbl, "schema_valid": False}
    return {"risk": None, "schema_valid": False}


def _parse_segmentation(text: str, labels: list) -> dict:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            seg = str(obj.get("segment", "")).strip()
            rationale = str(obj.get("rationale", obj.get("reason", ""))).strip()
            if seg in labels:
                return {"segment": seg, "rationale": rationale, "schema_valid": True}
        except Exception:
            pass
    upper = text.upper()
    for lbl in labels:
        if lbl.upper() in upper:
            return {"segment": lbl, "rationale": "", "schema_valid": False}
    return {"segment": None, "rationale": "", "schema_valid": False}


def _parse_support(text: str, labels: list) -> dict:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            intent = str(obj.get("intent", obj.get("category", ""))).strip()
            if intent in labels:
                return {"intent": intent, "schema_valid": True}
        except Exception:
            pass
    upper = text.upper()
    for lbl in labels:
        if lbl.upper() in upper:
            return {"intent": lbl, "schema_valid": False}
    return {"intent": None, "schema_valid": False}


def _parse_content(text: str, labels: list) -> dict:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            strat = str(obj.get("strategy", obj.get("content_strategy", ""))).strip()
            if strat in labels:
                return {"strategy": strat, "schema_valid": True}
        except Exception:
            pass
    upper = text.upper()
    for lbl in labels:
        if lbl.upper() in upper:
            return {"strategy": lbl, "schema_valid": False}
    return {"strategy": None, "schema_valid": False}


def _parse_recommendation(text: str, labels: list) -> dict:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            action = str(obj.get("action", obj.get("recommendation", ""))).strip()
            if action in labels:
                return {"action": action, "schema_valid": True}
        except Exception:
            pass
    upper = text.upper()
    for lbl in labels:
        if lbl.upper() in upper:
            return {"action": lbl, "schema_valid": False}
    return {"action": None, "schema_valid": False}


# ------------------------------------------------------------------
# Prompt templates per model type
# ------------------------------------------------------------------

_PROMPTS = {
    "churn": "Predict churn risk from customer telecom profile and reply with strict JSON.",
    "segmentation": "Classify this customer into exactly one marketing segment and reply with strict JSON.",
    "support": "Classify this customer support query into exactly one intent category and reply with strict JSON.",
    "content": "Recommend a content marketing strategy for this customer and reply with strict JSON.",
    "recommendation": "Determine a product recommendation action from this review and reply with strict JSON.",
}

_PARSERS = {
    "churn": _parse_churn,
    "segmentation": _parse_segmentation,
    "support": _parse_support,
    "content": _parse_content,
    "recommendation": _parse_recommendation,
}


# ------------------------------------------------------------------
# Public model getters
# ------------------------------------------------------------------

def get_sentiment_model():
    """Load the fine-tuned DistilBERT sentiment model."""
    key = "sentiment"
    with _lock:
        if key not in _cache:
            model_dir = ARTIFACTS_ROOT / "distilbert_sentiment_production" / "final_model"
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Sentiment model directory not found: {model_dir}. "
                    "Run ./scripts/setup_artifacts.sh to download the model artifacts."
                )
            wrapper = model_dir / "sentiment_agent.py"
            kwargs: Dict[str, Any] = {"model_path": str(model_dir)}
            device = _resolve_torch_device()
            if device:
                kwargs["device"] = device

            _cache[key] = _load_wrapper_from_file(wrapper, "SentimentAgent", **kwargs)
            logger.info("Loaded DistilBERT sentiment model from %s", model_dir)
        return _cache[key]


def get_churn_xgboost_model():
    """Load the fine-tuned XGBoost churn predictor."""
    key = "churn_xgb"
    with _lock:
        if key not in _cache:
            model_dir = ARTIFACTS_ROOT / "churn_xgboost_production" / "model"
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"XGBoost model directory not found: {model_dir}. "
                    f"Run 07_churn_xgboost_vs_llm.ipynb to generate it."
                )
            wrapper = model_dir / "churn_xgboost_predictor.py"
            _cache[key] = _load_wrapper_from_file(
                wrapper, "ChurnXGBoostPredictor", model_dir=str(model_dir)
            )
            logger.info("Loaded XGBoost churn model from %s", model_dir)
        return _cache[key]


def _load_lora_model(model_type: str, artifact_name: str):
    """Load a LoRA adapter on top of the shared GGUF base."""
    artifact_dir = ARTIFACTS_ROOT / artifact_name
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"Artifact directory not found: {artifact_dir}. "
            f"Run the corresponding fine-tuning notebook to generate it."
        )
    summary = _read_summary(artifact_dir)
    labels = summary.get("labels", [])
    adapter_path = str(artifact_dir / "final_model" / "lora_adapter")
    device = _resolve_torch_device() or "cpu"

    return _LoraInferenceWrapper(
        adapter_path=adapter_path,
        labels=labels,
        device=device,
        max_length=int(summary.get("max_length", 384)),
        max_new_tokens=int(summary.get("gen_max_new_tokens", 40)),
        prompt_template=_PROMPTS.get(model_type, ""),
        parse_fn=_PARSERS.get(model_type),
    )


def get_churn_llm_model():
    """Load the LoRA-adapted churn risk LLM."""
    key = "churn_llm"
    with _lock:
        if key not in _cache:
            _cache[key] = _load_lora_model("churn", "churn_llm_production")
            logger.info("Loaded churn LLM model")
        return _cache[key]


def get_segmentation_model():
    """Load the LoRA-adapted segmentation LLM (v2 robust)."""
    key = "segmentation"
    with _lock:
        if key not in _cache:
            _cache[key] = _load_lora_model("segmentation", "segmentation_llm_v2_robust")
            logger.info("Loaded segmentation LLM model")
        return _cache[key]


def get_support_model():
    """Load the LoRA-adapted support intent LLM."""
    key = "support"
    with _lock:
        if key not in _cache:
            _cache[key] = _load_lora_model("support", "support_llm_production")
            logger.info("Loaded support intent LLM model")
        return _cache[key]


def get_content_model():
    """Load the LoRA-adapted content strategy LLM."""
    key = "content"
    with _lock:
        if key not in _cache:
            _cache[key] = _load_lora_model("content", "content_llm_production")
            logger.info("Loaded content strategy LLM model")
        return _cache[key]


def get_recommendation_model():
    """Load the LoRA-adapted recommendation LLM."""
    key = "recommendation"
    with _lock:
        if key not in _cache:
            _cache[key] = _load_lora_model("recommendation", "recommendation_llm_production")
            logger.info("Loaded recommendation LLM model")
        return _cache[key]
