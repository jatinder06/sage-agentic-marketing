# Churn Risk LLM (LoRA) — Production Export

## Task
Churn risk prediction from customer telecom profile.

## Output Contract
Strict JSON: `{"risk": "<LABEL>"}`

## Labels
['HIGH_RISK', 'LOW_RISK']

## Results
- Baseline accuracy: 0.7727
- Fine-tuned accuracy: 0.7914
- Baseline macro F1: 0.4359
- Fine-tuned macro F1: 0.6947
- Schema valid rate: 1.0000
- Forgetting rate: 14.71%

## Artifacts
- Adapter: `final_model/lora_adapter/`
- Wrapper: `final_model/churn_agent_llm.py`
- Integration stubs: `integration/`
