# DistilBERT Sentiment Classifier — Marketing Domain

## Model Details
- **Base model**: distilbert-base-uncased
- **Task**: 3-class sentiment classification (negative, neutral, positive)
- **Domain**: Product/marketing reviews
- **Fine-tuned on**: 57,711 training samples

## Performance (Test Set: 8,682 samples)
| Metric | Value |
|--------|-------|
| Accuracy | 0.7985 |
| F1 (weighted) | 0.7980 |
| ROC-AUC | 0.9318531430052228 |
| ECE | 0.0770 |
| Forgetting Rate | 8.19% |

## Per-Class F1
| Class | F1 |
|-------|----|
| negative | 0.7922 |
| neutral | 0.7021 |
| positive | 0.8942 |


## Usage in Multi-Agentic System
```python
from sentiment_agent import SentimentAgent

agent = SentimentAgent("/Users/jatindersingh/work_data/mkt-ai/experimentation/artifacts/distilbert_sentiment_production/final_model")
result = agent.predict("Great product, highly recommend!")
# {'label': 'positive', 'confidence': 0.95, 'scores': {...}}
```

## Training Config
- Epochs: 6.0 (early stopping patience=3)
- LR: 2e-05, warmup: 0.1
- Loss: Weighted CE
- Gradient clipping: 1.0
