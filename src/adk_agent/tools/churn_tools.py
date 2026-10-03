"""Churn prediction tools using fine-tuned XGBoost and LoRA-adapted LLM.

XGBoost model: artifacts/churn_xgboost_production/
  - 36 engineered features, Optuna-tuned, Accuracy: 74.0%, ROC-AUC: 0.846

Churn LLM: artifacts/churn_llm_production/
  - Qwen2.5-0.5B + LoRA, text-based profile input, Accuracy: 79.1%
"""

from typing import Dict, Any

import pandas as pd

from src.adk_agent.tools.model_client import predict


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the 17 engineered features from notebook 07."""
    out = df.copy()

    out["AvgChargesPerMonth"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
    out["ChargeGrowthRate"] = (
        (out["MonthlyCharges"] - out["AvgChargesPerMonth"])
        / out["AvgChargesPerMonth"].clip(lower=0.01)
    )
    out["TenureMonthlyInteraction"] = out["tenure"] * out["MonthlyCharges"]
    out["TenureBin"] = pd.cut(
        out["tenure"],
        bins=[-1, 6, 12, 24, 48, 72, 9999],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(int)

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    out["ServiceCount"] = sum(
        (out[c].isin(["Yes", "Yes, but no internet"])).astype(int)
        for c in service_cols
    )

    protection_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    out["ProtectionCount"] = sum(
        (out[c] == "Yes").astype(int) for c in protection_cols
    )
    out["HasProtectionBundle"] = (out["ProtectionCount"] >= 3).astype(int)

    streaming_cols = ["StreamingTV", "StreamingMovies"]
    out["StreamingCount"] = sum(
        (out[c] == "Yes").astype(int) for c in streaming_cols
    )

    out["CostPerService"] = out["MonthlyCharges"] / out["ServiceCount"].clip(lower=1)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    out["ContractRisk"] = out["Contract"].map(contract_map).fillna(0).astype(int)

    out["IsElectronicCheck"] = (out["PaymentMethod"] == "Electronic check").astype(int)
    out["IsAutoPayment"] = (
        out["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
    )

    out["ShortTenureMonthly"] = (
        (out["tenure"] <= 12) & (out["ContractRisk"] == 0)
    ).astype(int)

    out["IsFiberOptic"] = (out["InternetService"] == "Fiber optic").astype(int)
    out["FiberNoProtection"] = (
        (out["IsFiberOptic"] == 1) & (out["ProtectionCount"] == 0)
    ).astype(int)

    out["HasPartnerOrDependents"] = (
        (out["Partner"] == "Yes") | (out["Dependents"] == "Yes")
    ).astype(int)
    out["SeniorAlone"] = (
        (out["SeniorCitizen"] == 1) & (out["HasPartnerOrDependents"] == 0)
    ).astype(int)

    return out


def _encode_and_scale(df: pd.DataFrame, label_encoders: dict, scaler) -> pd.DataFrame:
    """Apply saved label encoders and standard scaler."""
    out = df.copy()
    for col, le in label_encoders.items():
        if col in out.columns:
            out[col] = out[col].astype(str).map(
                {cls: idx for idx, cls in enumerate(le.classes_)}
            )
            out[col] = out[col].fillna(0).astype(int)

    scale_cols = [c for c in scaler.feature_names_in_ if c in out.columns]
    if scale_cols:
        out[scale_cols] = scaler.transform(out[scale_cols])

    return out


def predict_churn_xgboost(
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    contract: str,
    internet_service: str,
    payment_method: str,
    gender: str = "Male",
    senior_citizen: int = 0,
    partner: str = "No",
    dependents: str = "No",
    phone_service: str = "Yes",
    multiple_lines: str = "No",
    online_security: str = "No",
    online_backup: str = "No",
    device_protection: str = "No",
    tech_support: str = "No",
    streaming_tv: str = "No",
    streaming_movies: str = "No",
    paperless_billing: str = "Yes",
) -> Dict[str, Any]:
    """Predict customer churn risk using the XGBoost model with structured features.

    Args:
        tenure: Months the customer has been with the company.
        monthly_charges: Current monthly charge amount.
        total_charges: Total charges over the customer lifetime.
        contract: Contract type (Month-to-month, One year, Two year).
        internet_service: Internet service type (DSL, Fiber optic, No).
        payment_method: Payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
        gender: Customer gender (Male, Female).
        senior_citizen: Whether customer is a senior citizen (0 or 1).
        partner: Has partner (Yes, No).
        dependents: Has dependents (Yes, No).
        phone_service: Has phone service (Yes, No).
        multiple_lines: Has multiple lines (Yes, No, No phone service).
        online_security: Has online security (Yes, No, No internet service).
        online_backup: Has online backup (Yes, No, No internet service).
        device_protection: Has device protection (Yes, No, No internet service).
        tech_support: Has tech support (Yes, No, No internet service).
        streaming_tv: Has streaming TV (Yes, No, No internet service).
        streaming_movies: Has streaming movies (Yes, No, No internet service).
        paperless_billing: Has paperless billing (Yes, No).

    Returns:
        Dictionary with risk label (HIGH_RISK/LOW_RISK) and churn probability.
    """
    try:
        result = predict("/predict/churn_xgboost", {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "paperless_billing": paperless_billing,
        })

        return {
            "status": "success",
            "risk": result["risk"],
            "churn_probability": result["churn_probability"],
            "model": "xgboost-churn-optuna",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def predict_churn_llm(customer_profile: str) -> Dict[str, Any]:
    """Predict churn risk from a natural-language customer profile using the fine-tuned LLM.

    Args:
        customer_profile: Text description of the customer, e.g.
            'tenure=59 months; contract=One year; internet=Fiber optic;
             monthly_charges=85.5; online_security=Yes; tech_support=No'

    Returns:
        Dictionary with risk label and schema validity flag.
    """
    try:
        if not customer_profile or not customer_profile.strip():
            return {"status": "error", "error": "Customer profile text is empty."}

        result = predict("/predict/churn_llm", {"text": customer_profile.strip()})

        return {
            "status": "success" if result["success"] else "error",
            "risk": result.get("risk"),
            "schema_valid": result.get("schema_valid", False),
            "model": "qwen2.5-churn-lora",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
