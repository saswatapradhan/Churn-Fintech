"""
Batch Scoring Pipeline — PayPal SMB EU Churn
Scores the entire SMB account portfolio in one vectorized pass (not one-by-one
via the API). This is what a nightly Airflow/cron job would run in production,
and its output feeds both the business dashboard and prediction logging.
"""
import pandas as pd
import joblib
import json
from datetime import datetime, timezone
from xgboost import XGBClassifier
from src.features.build_features import add_derived_features

RAW_PATH = "data/raw/paypal_smb_eu_churn_raw.csv"
MODEL_PATH = "src/serving/model/model.xgb"
PREPROCESSOR_PATH = "src/serving/model/preprocessing.pkl"
OUTPUT_PATH = "data/processed/scored_accounts.csv"

CATEGORICAL_COLS = ["country", "industry"]
NUMERIC_COLS = [
    "account_age_months", "monthly_tpv", "tpv_trend_3m_pct",
    "txn_count_monthly", "avg_ticket_size", "num_products_used",
    "login_freq_monthly", "disputes_90d", "chargebacks_90d",
    "support_tickets_90d", "failed_txn_ratio"
]

def load_artifacts():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

def score_portfolio() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    model, preprocessor = load_artifacts()

    X_raw = df[CATEGORICAL_COLS + NUMERIC_COLS]
    X_transformed = preprocessor.transform(X_raw)

    feature_names = (
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS))
        + NUMERIC_COLS
    )
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    X_df = add_derived_features(X_df)

    probabilities = model.predict_proba(X_df)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results = pd.DataFrame({
        "account_id": df["account_id"],
        "country": df["country"],
        "industry": df["industry"],
        "monthly_tpv": df["monthly_tpv"],
        "churn_probability": probabilities.round(4),
        "churn_prediction": predictions,
        "risk_level": pd.cut(
            probabilities, bins=[-0.01, 0.3, 0.6, 1.01],
            labels=["LOW", "MEDIUM", "HIGH"]
        ),
        "priority_score": (probabilities * df["monthly_tpv"]).round(2),
        "scored_at": datetime.now(timezone.utc).isoformat()
    })

    results = results.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return results

if __name__ == "__main__":
    scored_df = score_portfolio()
    scored_df.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "total_accounts_scored": len(scored_df),
        "high_risk_count": int((scored_df["risk_level"] == "HIGH").sum()),
        "medium_risk_count": int((scored_df["risk_level"] == "MEDIUM").sum()),
        "low_risk_count": int((scored_df["risk_level"] == "LOW").sum()),
        "total_tpv_at_high_risk": float(
            scored_df.loc[scored_df["risk_level"] == "HIGH", "monthly_tpv"].sum()
        ),
        "scored_at": scored_df["scored_at"].iloc[0]
    }

    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nTop 5 priority accounts for Sales outreach:")
    print(scored_df[["account_id", "country", "industry", "churn_probability",
                      "priority_score", "risk_level"]].head())