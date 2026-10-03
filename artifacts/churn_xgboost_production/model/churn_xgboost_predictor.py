"""XGBoost Churn Predictor wrapper for multi-agent integration."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib


class ChurnXGBoostPredictor:
    """Production inference wrapper for XGBoost churn model."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_dir / "xgboost_churn.json"))
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.label_encoders = joblib.load(model_dir / "label_encoders.joblib")
        self.feature_names = joblib.load(model_dir / "feature_names.joblib")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([features])
        prob = float(self.model.predict_proba(df[self.feature_names])[:, 1][0])
        label = "HIGH_RISK" if prob >= 0.5 else "LOW_RISK"
        return {
            "success": True,
            "risk": label,
            "churn_probability": round(prob, 4),
        }

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(f) for f in features_list]
