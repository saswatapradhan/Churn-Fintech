"""Model inference logic used by FastAPI. (Serving step)"""
"""
Inference logic — loads the trained model + preprocessor once,
and exposes a single function to score new SMB account data.
"""
import joblib
import pandas as pd
from xgboost import XGBClassifier
from src.utils.prediction_logger import log_prediction

MODEL_PATH = "src/serving/model/model.xgb"
PREPROCESSOR_PATH = "src/serving/model/preprocessing.pkl"

CATEGORICAL_COLS = ["country", "industry"]
NUMERIC_COLS = [
    "account_age_months", "monthly_tpv", "tpv_trend_3m_pct",
    "txn_count_monthly", "avg_ticket_size", "num_products_used",
    "login_freq_monthly", "disputes_90d", "chargebacks_90d",
    "support_tickets_90d", "failed_txn_ratio"
]

_model = None
_preprocessor = None

def load_artifacts():
    global _model, _preprocessor
    if _model is None:
        _model = XGBClassifier()
        _model.load_model(MODEL_PATH)
    if _preprocessor is None:
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
    return _model, _preprocessor

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dispute_to_txn_ratio"] = df["disputes_90d"] / df["txn_count_monthly"].replace(0, 1)
    df["chargeback_to_txn_ratio"] = df["chargebacks_90d"] / df["txn_count_monthly"].replace(0, 1)
    df["engagement_per_product"] = df["login_freq_monthly"] / df["num_products_used"].replace(0, 1)
    df["is_sharp_decline"] = (df["tpv_trend_3m_pct"] < -30).astype(int)
    return df

def predict_churn(record: dict,source: str = "api") -> dict:
    model, preprocessor = load_artifacts()

    df = pd.DataFrame([record])
    X_raw = df[CATEGORICAL_COLS + NUMERIC_COLS]
    X_transformed = preprocessor.transform(X_raw)

    feature_names = (
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS))
        + NUMERIC_COLS
    )
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    X_df = add_derived_features(X_df)  # X_df already has all NUMERIC_COLS needed

    proba = model.predict_proba(X_df)[0][1]
    prediction = int(proba >= 0.5)

    result = {
        "churn_prediction": prediction,
        "churn_probability": round(float(proba), 4),
        "risk_level": "HIGH" if proba >= 0.6 else "MEDIUM" if proba >= 0.3 else "LOW"
    }

    log_prediction(record, result, source=source)  # ADD THIS LINE

    return result


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Defensive coercion — callers (Gradio, FastAPI, batch jobs) may pass
    # numeric-looking strings; force them to numeric before any arithmetic.
    numeric_cols_used = ["disputes_90d", "chargebacks_90d", "txn_count_monthly",
                          "login_freq_monthly", "num_products_used", "tpv_trend_3m_pct"]
    for col in numeric_cols_used:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["dispute_to_txn_ratio"] = df["disputes_90d"] / df["txn_count_monthly"].replace(0, 1)
    df["chargeback_to_txn_ratio"] = df["chargebacks_90d"] / df["txn_count_monthly"].replace(0, 1)
    df["engagement_per_product"] = df["login_freq_monthly"] / df["num_products_used"].replace(0, 1)
    df["is_sharp_decline"] = (df["tpv_trend_3m_pct"] < -30).astype(int)

    return df
    
if __name__ == "__main__":
    sample = {
        "account_id": "SMB_EU_TEST_001",
        "country": "DE",
        "industry": "E-commerce",
        "account_age_months": 8,
        "monthly_tpv": 4500,
        "tpv_trend_3m_pct": -45.0,
        "txn_count_monthly": 30,
        "avg_ticket_size": 150,
        "num_products_used": 1,
        "login_freq_monthly": 2,
        "disputes_90d": 4,
        "chargebacks_90d": 2,
        "support_tickets_90d": 3,
        "failed_txn_ratio": 0.25
    }
    result = predict_churn(sample)
    print(result)