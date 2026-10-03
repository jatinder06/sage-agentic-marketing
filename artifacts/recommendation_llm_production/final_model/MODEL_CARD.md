# Recommendation LLM (LoRA) — Production Fine-Tuning

## Task
Product recommendation classification from review text into 3 action categories.
The model receives natural-language product reviews and outputs structured JSON
classification for downstream multi-agent integration.

## Output Contract
Strict JSON: `{"action": "<LABEL>"}`

## Labels
['CONSIDER', 'DO_NOT_RECOMMEND', 'RECOMMEND']

## Base Model
Qwen2.5-0.5B-Instruct (494M parameters, loaded from local GGUF)

## v1 vs v2 Comparison

| Metric | v1 (Experimental) | v2 (Production) |
|---|---|---|
| Accuracy | Not measured comprehensively | 74.0% |
| Macro F1 | Not measured comprehensively | 73.5% |
| General Capability | Not measured | 138% retained |
| Noise Robustness (50% trunc) | Not tested | 70.0% |
| Trainable Params | All candidate modules | 1,081,344 (2 modules) |
| Schema Valid Rate | Regex-based | 100.0% |
| Output Format | Free-text ACTION+RATIONALE | Strict JSON |

## Fixes Applied in v2
1. **Output contract**: Free-text -> strict JSON for reliable parsing
2. **Learning rate**: 2e-4 -> 5e-5 (4x lower)
3. **LoRA scope**: All candidate modules -> 2 (q_proj + v_proj only)
4. **Replay buffer**: 400 general-instruction examples
5. **Data augmentation**: Text truncation + 3 prompt templates x 3 instruction variants
6. **Regularization**: weight_decay 0.01 -> 0.05, LoRA dropout 0.05 -> 0.1

## sklearn Baseline Context
TF-IDF + classifiers provide accuracy on structured text features.
The LLM adds value for: (a) diverse input formats, (b) structured JSON output,
(c) multi-agent integration, (d) generalization to unseen text patterns.

## Artifacts
- LoRA adapter: `final_model/lora_adapter/`
- Inference wrapper: `final_model/recommendation_agent_llm.py`
- Integration stubs: `integration/`
- Experiment metrics: `experiment_summary.json`
