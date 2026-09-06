"""Load raw PayPal SMB EU churn data. (To be implemented in the Preprocessing step)"""

"""Load raw PayPal SMB EU churn data."""
import pandas as pd

RAW_PATH = "data/raw/paypal_smb_eu_churn_raw.csv"

def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())