# Support Intent LLM (LoRA) — Production Export

## Task
Customer support intent classification from query text.

## Output Contract
Strict JSON: `{"tag": "<LABEL>"}`

## Labels
['ACCOUNT', 'BILLING', 'DELIVERY', 'GENERAL', 'TECHNICAL']

## Results
- Baseline accuracy: 0.3273
- Fine-tuned accuracy: 0.9471
- Baseline macro F1: 0.2251
- Fine-tuned macro F1: 0.9495
- Schema valid rate: 1.0000
- Forgetting rate: 79.17%

## Artifacts
- Adapter: `final_model/lora_adapter/`
- Wrapper: `final_model/support_agent_llm.py`
- Integration stubs: `integration/`
