"""
Prediction Logging — PayPal SMB EU Churn
Appends every prediction (API, Gradio, or batch) to a persistent log.
Writes both locally (best-effort, for local debugging) and to S3 (durable,
shared across the API service, ECS, and both dashboards).
"""
import pandas as pd
import os
from datetime import datetime, timezone

LOG_PATH = "data/logs/predictions_log.csv"
S3_LOG_KEY = "logs/predictions_log.csv"

LOG_COLUMNS = [
    "logged_at", "account_id", "source", "model_version",
    "country", "industry", "account_age_months", "monthly_tpv",
    "tpv_trend_3m_pct", "txn_count_monthly", "avg_ticket_size",
    "num_products_used", "login_freq_monthly", "disputes_90d",
    "chargebacks_90d", "support_tickets_90d", "failed_txn_ratio",
    "churn_probability", "churn_prediction", "risk_level"
]

def log_prediction(record: dict, result: dict, source: str, model_version: str = "v1_tuned"):
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

    # Local write — best-effort, doesn't fail the request if it errors
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if os.path.exists(LOG_PATH):
            df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
        else:
            df_row.to_csv(LOG_PATH, mode="w", header=True, index=False)
    except Exception as e:
        print(f"Warning: local prediction log write failed: {e}")

    # S3 write — durable, shared across every service that reads this log
    try:
        from src.utils.s3_helper import file_exists_in_s3, download_dataframe_from_s3, upload_dataframe_to_s3
        if file_exists_in_s3(S3_LOG_KEY):
            existing = download_dataframe_from_s3(S3_LOG_KEY)
            combined = pd.concat([existing, df_row], ignore_index=True)
        else:
            combined = df_row
        upload_dataframe_to_s3(combined, S3_LOG_KEY)
    except Exception as e:
        print(f"Warning: S3 prediction log write failed: {e}")

def read_log() -> pd.DataFrame:
    """Reads from S3 first (the durable, shared copy); falls back to local file."""
    try:
        from src.utils.s3_helper import file_exists_in_s3, download_dataframe_from_s3
        if file_exists_in_s3(S3_LOG_KEY):
            return download_dataframe_from_s3(S3_LOG_KEY)
    except Exception as e:
        print(f"Warning: could not read S3 prediction log: {e}")

    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)