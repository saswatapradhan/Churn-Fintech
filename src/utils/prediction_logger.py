"""
Prediction Logging — PayPal SMB EU Churn
Appends every prediction (API, Gradio, or batch) to a persistent log.
This log is the data source for drift detection, the model health dashboard,
and eventually the feedback loop (Step 7).
"""
import pandas as pd
import os
from datetime import datetime, timezone

LOG_PATH = "data/logs/predictions_log.csv"

LOG_COLUMNS = [
    "logged_at", "account_id", "source", "model_version",
    "country", "industry", "account_age_months", "monthly_tpv",
    "tpv_trend_3m_pct", "txn_count_monthly", "avg_ticket_size",
    "num_products_used", "login_freq_monthly", "disputes_90d",
    "chargebacks_90d", "support_tickets_90d", "failed_txn_ratio",
    "churn_probability", "churn_prediction", "risk_level"
]

def log_prediction(record: dict, result: dict, source: str, model_version: str = "v1_tuned"):
    """
    Appends one prediction to the log. Called after every predict_churn() call.
    source: 'api', 'gradio', or 'batch' — tells us where the prediction came from.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "account_id": record.get("account_id"),
        "source": source,
        "model_version": model_version,
        **{k: record.get(k) for k in [
            "country", "industry", "account_age_months", "monthly_tpv",
            "tpv_trend_3m_pct", "txn_count_monthly", "avg_ticket_size",
            "num_products_used", "login_freq_monthly", "disputes_90d",
            "chargebacks_90d", "support_tickets_90d", "failed_txn_ratio"
        ]},
        "churn_probability": result.get("churn_probability"),
        "churn_prediction": result.get("churn_prediction"),
        "risk_level": result.get("risk_level"),
    }

    df_row = pd.DataFrame([row], columns=LOG_COLUMNS)

    if os.path.exists(LOG_PATH):
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode="w", header=True, index=False)

def read_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_PATH)