# Segmentation LLM v2 (LoRA) — Robust Fine-Tuning

## Task
Customer segmentation from RFM-style feature prompts into 4 marketing segments.
The model receives natural-language descriptions of customer metrics and outputs
structured JSON classification.

## Output Contract
Strict JSON: `{"segment": "<LABEL>"}`

## Labels
['At Risk', 'Champions', 'Loyal', 'Needs Attention']

## Base Model
Qwen2.5-0.5B-Instruct (494M parameters, loaded from local GGUF)

## v1 vs v2 Comparison

| Metric | v1 | v2 |
|---|---|---|
| Accuracy | 100.0% | 91.2% |
| Macro F1 | 100.0% | 91.6% |
| General Capability | Not measured (76% forgetting) | 163% retained (improved) |
| Noise Robustness (15%) | Not measured | 85.5% |
| Trainable Params | 8.8M (7 modules) | 1,081,344 (2 modules) |
| Schema Valid Rate | 100% | 100% |

## Fixes Applied in v2
1. **Learning rate**: 2e-4 → 5e-5 (4x lower — gentler weight updates)
2. **LoRA scope**: 7 projection modules → 2 (q_proj + v_proj only, 8x fewer trainable params)
3. **Replay buffer**: 400 general-instruction examples (math, knowledge, instructions, JSON)
4. **Data augmentation**: ±8% numeric jitter + 3 prompt templates x 3 instruction variants
5. **Regularization**: weight_decay 0.01 → 0.05, LoRA dropout 0.05 → 0.1

## Key Finding
v1 achieved 100% accuracy but suffered catastrophic forgetting (76% of general knowledge destroyed).
v2 achieves 91.2% accuracy while **improving** general capability by 63% (48.5% → 79.0%).
The replay buffer and reduced LoRA scope successfully prevent catastrophic forgetting.

## sklearn Baseline Context
DecisionTree/RandomForest/GradientBoosting all achieve 100% accuracy on structured
features because segments are a deterministic function of RFM_Score (assigned via pd.cut
with fixed bins). The LLM's value is in handling natural-language inputs, multiple prompt
formats, and generalising to new schemas without retraining.

## Artifacts
- LoRA adapter: `final_model/lora_adapter/`
- Inference wrapper: `final_model/segmentation_agent_llm.py`
- Integration stubs: `integration/`
- Experiment metrics: `experiment_summary.json`
