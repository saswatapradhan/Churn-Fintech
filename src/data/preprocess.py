"""Cleaning + encoding logic for PayPal SMB EU churn data. (Preprocessing step)"""

"""
Preprocessing — PayPal SMB EU Churn
Uses a scikit-learn ColumnTransformer (not pd.get_dummies) so the exact same
encoder object can be pickled and reused at inference time (train/serve parity).
"""
import pandas as pd
import joblib
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = ["country", "industry"]
ID_COL = "account_id"
TARGET_COL = "churn"

NUMERIC_COLS = [
    "account_age_months", "monthly_tpv", "tpv_trend_3m_pct",
    "txn_count_monthly", "avg_ticket_size", "num_products_used",
    "login_freq_monthly", "disputes_90d", "chargebacks_90d",
    "support_tickets_90d", "failed_txn_ratio"
]

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )

def preprocess(df: pd.DataFrame, fit: bool = True, preprocessor: ColumnTransformer = None):
    df = df.dropna().drop_duplicates(subset=[ID_COL]).reset_index(drop=True)

    ids = df[ID_COL]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    X_raw = df[CATEGORICAL_COLS + NUMERIC_COLS]

    if fit:
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X_raw)
    else:
        X_transformed = preprocessor.transform(X_raw)

    feature_names = (
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS))
        + NUMERIC_COLS
    )
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    result = pd.concat([ids.reset_index(drop=True), X_df], axis=1)
    if y is not None:
        result[TARGET_COL] = y.reset_index(drop=True)

    return result, preprocessor, feature_names

if __name__ == "__main__":
    from src.data.load_data import load_raw_data

    df = load_raw_data()
    processed_df, fitted_preprocessor, feature_cols = preprocess(df, fit=True)

    processed_df.to_csv("data/processed/paypal_smb_eu_churn_processed.csv", index=False)
    joblib.dump(fitted_preprocessor, "src/serving/model/preprocessing.pkl")

    with open("config/feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Processed shape: {processed_df.shape}")
    print("Saved: data/processed/paypal_smb_eu_churn_processed.csv")
    print("Saved: src/serving/model/preprocessing.pkl")
    print("Saved: config/feature_columns.json")