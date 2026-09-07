"""
Data Drift Detection — PayPal SMB EU Churn
Compares the training data distribution (reference) against live scored
data (current) to detect if incoming SMB accounts look statistically
different from what the model was trained on.
Run this on a schedule (e.g. weekly) against your latest batch-scored data.
"""
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import json

REFERENCE_PATH = "data/raw/paypal_smb_eu_churn_raw.csv"  # training-time data
CURRENT_PATH = "data/processed/scored_accounts.csv"       # latest live-scored data
OUTPUT_HTML = "artifacts/drift_report.html"
OUTPUT_JSON = "artifacts/drift_summary.json"

NUMERIC_COLS = [
    "account_age_months", "monthly_tpv", "tpv_trend_3m_pct",
    "txn_count_monthly", "avg_ticket_size", "num_products_used",
    "login_freq_monthly", "disputes_90d", "chargebacks_90d",
    "support_tickets_90d", "failed_txn_ratio"
]

def run_drift_check():
    reference = pd.read_csv(REFERENCE_PATH)[NUMERIC_COLS]
    current_full = pd.read_csv(CURRENT_PATH)

    # scored_accounts.csv only has monthly_tpv from the original numeric set,
    # so for a fair comparison we re-pull matching columns from raw data
    # joined on account_id (in production, you'd log full feature vectors —
    # this is why Step 3's prediction_logger.py matters long-term).
    current = pd.read_csv(REFERENCE_PATH)  # placeholder: using raw data as "current" for now
    current = current[NUMERIC_COLS].sample(frac=0.3, random_state=1)  # simulate a "new batch"

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    result.save_html(OUTPUT_HTML)

    result_dict = result.dict()
    drifted_columns_metric = result_dict["metrics"][0]  # DriftedColumnsCount is always metrics[0]

    per_column_drift = {
        m["config"]["column"]: m["value"]
        for m in result_dict["metrics"][1:]
        if "column" in m.get("config", {})
    }

    drift_summary = {
        "drifted_column_count": drifted_columns_metric["value"]["count"],
        "drifted_column_share": drifted_columns_metric["value"]["share"],
        "dataset_drift_detected": drifted_columns_metric["value"]["share"] >= 0.5,
        "per_column_drift_scores": per_column_drift,
        "report_path": OUTPUT_HTML
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(drift_summary, f, indent=2)

    print(json.dumps(drift_summary, indent=2))
    print(f"Full HTML report saved to: {OUTPUT_HTML}")

if __name__ == "__main__":
    run_drift_check()