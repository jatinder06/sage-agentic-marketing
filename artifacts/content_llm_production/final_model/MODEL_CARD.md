# Content Strategy LLM (LoRA) — Production Export

## Task
Campaign content strategy classification from customer marketing profile.

## Output Contract
Strict JSON: `{"strategy": "<LABEL>"}`

## Labels
['DISCOUNT_REENGAGE', 'EDUCATIONAL_NURTURE', 'LOYALTY_UPSELL']

## Results
- Baseline accuracy: 0.4409
- Fine-tuned accuracy: 0.9820
- Baseline macro F1: 0.2040
- Fine-tuned macro F1: 0.9590
- Schema valid rate: 1.0000
- Forgetting rate: 55.67%

## Artifacts
- Adapter: `final_model/lora_adapter/`
- Wrapper: `final_model/content_agent_llm.py`
- Integration stubs: `integration/`
