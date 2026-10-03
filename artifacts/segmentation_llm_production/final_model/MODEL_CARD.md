# Segmentation LLM (LoRA) — Production Export

## Task
Customer segmentation from RFM-style feature prompt.

## Output Contract
Strict JSON: `{"segment": "<LABEL>"}`

## Labels
['At Risk', 'Champions', 'Loyal', 'Needs Attention']

## Results
- Baseline accuracy: 0.2273
- Fine-tuned accuracy: 1.0000
- Baseline macro F1: 0.0926
- Fine-tuned macro F1: 1.0000
- Schema valid rate: 1.0000
- Forgetting rate: 76.00%

## Artifacts
- Adapter: `final_model/lora_adapter/`
- Wrapper: `final_model/segmentation_agent_llm.py`
- Integration stubs: `integration/`
