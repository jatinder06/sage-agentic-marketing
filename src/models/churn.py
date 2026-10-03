"""Classical churn-model metrics."""
from sklearn.metrics import roc_auc_score


def binary_auc(y_true, probabilities):
    return float(roc_auc_score(y_true, probabilities))