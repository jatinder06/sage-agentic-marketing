"""Shared validation for LoRA training notebooks."""
from pathlib import Path


def require_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path