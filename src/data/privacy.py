"""Privacy helpers used by preprocessing notebooks."""
import re


def mask_pii(value: object) -> str:
    text = str(value)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
    text = re.sub(r"\+?\d[\d ()-]{7,}\d", "[PHONE]", text)
    return re.sub(r"\b\d{8,}\b", "[ID]", text)