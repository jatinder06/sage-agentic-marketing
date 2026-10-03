"""Sentiment analysis inference wrapper for multi-agentic system integration."""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAgent:
    """Production sentiment classifier for agentic pipelines."""

    LABEL_NAMES = ["negative", "neutral", "positive"]

    def __init__(self, model_path: str, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=os.getenv("HF_HOME"))
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=os.getenv("HF_HOME"))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Single text prediction with confidence scores."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=256, padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Remove token_type_ids as DistilBERT doesn't use it
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        return {
            "label": self.LABEL_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "scores": {name: float(probs[i]) for i, name in enumerate(self.LABEL_NAMES)},
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Batch prediction for throughput."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=256, padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Remove token_type_ids as DistilBERT doesn't use it
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            for j, p in enumerate(probs):
                pred_idx = int(p.argmax())
                results.append({
                    "label": self.LABEL_NAMES[pred_idx],
                    "confidence": float(p[pred_idx]),
                    "scores": {name: float(p[k]) for k, name in enumerate(self.LABEL_NAMES)},
                })
        return results

    def __call__(self, text: str) -> dict:
        return self.predict(text)
