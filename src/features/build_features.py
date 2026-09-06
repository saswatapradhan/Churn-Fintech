"""Feature engineering for PayPal SMB EU churn model. (Feature Engineering step)"""
"""
Feature Engineering — PayPal SMB EU Churn
Takes the preprocessed (encoded) dataframe and adds any derived features
beyond raw encoding. Kept separate from preprocess.py so feature logic
can evolve independently of encoding/schema logic.
"""
import pandas as pd

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ratio features — often more predictive than raw counts
    df["dispute_to_txn_ratio"] = df["disputes_90d"] / df["txn_count_monthly"].replace(0, 1)
    df["chargeback_to_txn_ratio"] = df["chargebacks_90d"] / df["txn_count_monthly"].replace(0, 1)

    # Engagement per product — low engagement despite many products = disengaged power user
    df["engagement_per_product"] = df["login_freq_monthly"] / df["num_products_used"].replace(0, 1)

    # Flag: sharply declining account (business-rule feature, not just raw trend)
    df["is_sharp_decline"] = (df["tpv_trend_3m_pct"] < -30).astype(int)

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/paypal_smb_eu_churn_processed.csv")
    df_features = add_derived_features(df)
    df_features.to_csv("data/processed/paypal_smb_eu_churn_features.csv", index=False)
    print(f"Shape after feature engineering: {df_features.shape}")
    print(df_features[["dispute_to_txn_ratio", "chargeback_to_txn_ratio",
                        "engagement_per_product", "is_sharp_decline"]].describe())