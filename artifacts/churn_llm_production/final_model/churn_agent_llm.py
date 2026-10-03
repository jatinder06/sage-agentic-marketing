"""Churn Risk LLM wrapper for multi-agent integration."""

import json
import re
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class ChurnRiskLLMAgent:
    """Inference wrapper for LoRA-adapted churn risk model."""

    def __init__(
        self,
        base_model_ref: str,
        adapter_path: str,
        labels: List[str],
        gguf_file: str | None = None,
        device: str | None = None,
        max_length: int = 384,
        max_new_tokens: int = 40,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.labels = labels
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        tok_kwargs: Dict[str, Any] = {"use_fast": True}
        model_kwargs: Dict[str, Any] = {
            "dtype": torch.float16 if device == "cuda" else torch.float32
        }
        if gguf_file is not None:
            tok_kwargs["gguf_file"] = gguf_file
            model_kwargs["gguf_file"] = gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_ref, **tok_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(base_model_ref, **model_kwargs)
        model = PeftModel.from_pretrained(base, adapter_path)
        self.model = model.to(self.device)
        self.model.eval()

    @staticmethod
    def _format_prompt(input_text: str) -> str:
        instruction = "Predict churn risk from customer telecom profile and reply with strict JSON."
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )

    def _parse(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            raw = match.group(0)
            try:
                obj = json.loads(raw)
                risk = str(obj.get("risk", "")).strip()
                if risk in self.labels:
                    return {"risk": risk, "schema_valid": True}
            except Exception:
                pass

        upper = text.upper()
        for lbl in self.labels:
            if lbl.upper() in upper:
                return {"risk": lbl, "schema_valid": False}

        return {"risk": None, "schema_valid": False}

    def predict(self, input_text: str) -> Dict[str, Any]:
        prompt = self._format_prompt(input_text)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
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
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        parsed = self._parse(text)

        return {
            "success": parsed["risk"] is not None,
            "risk": parsed["risk"],
            "schema_valid": parsed["schema_valid"],
            "raw_output": text,
        }

    def predict_batch(self, inputs: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(x) for x in inputs]
