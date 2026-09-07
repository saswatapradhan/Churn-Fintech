"""
Feedback Loop — PayPal SMB EU Churn
Compares predictions made 90+ days ago against actual outcomes, computing
REALIZED precision/recall — the metric that matters in production, separate
from training-time metrics which only measure historical performance.

Real usage: run this on a schedule (e.g. monthly). It only evaluates
predictions old enough to have a known outcome; everything else is skipped
until it ages in.
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score

LOG_PATH = "data/logs/predictions_log.csv"
OUTPUT_PATH = "artifacts/realized_performance.json"
FEEDBACK_WINDOW_DAYS = 90

def get_eligible_predictions() -> pd.DataFrame:
    """Real logic: only predictions old enough to have a known outcome."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()

    log_df = pd.read_csv(LOG_PATH)
    log_df["logged_at"] = pd.to_datetime(log_df["logged_at"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=FEEDBACK_WINDOW_DAYS)
    eligible = log_df[log_df["logged_at"] <= cutoff]
    return eligible

def simulate_ground_truth(eligible_df: pd.DataFrame) -> pd.DataFrame:
    """
    SIMULATED — stands in for a real outcomes feed (e.g. a query against
    account activity 90 days after prediction). Uses the same churn-driving
    signals as the original dataset generator, with noise, so the "realized"
    metrics look plausible without claiming to be real business results.
    """
    np.random.seed(7)
    z = (
        -0.045 * eligible_df["tpv_trend_3m_pct"]
        + 0.35 * eligible_df["disputes_90d"]
        + 0.30 * eligible_df["chargebacks_90d"]
        - 0.30 * eligible_df["num_products_used"]
        + np.random.normal(0, 1.2, len(eligible_df))
        - 1.0
    )
    prob_actual_churn = 1 / (1 + np.exp(-z))
    eligible_df = eligible_df.copy()
    eligible_df["actual_churn"] = (prob_actual_churn > np.random.uniform(0.5, 0.75, len(eligible_df))).astype(int)
    return eligible_df

def run_feedback_loop(use_simulation_if_empty: bool = True):
    eligible = get_eligible_predictions()

    is_simulated = False
    if eligible.empty and use_simulation_if_empty:
        print("No predictions are 90+ days old yet — using SIMULATED ground truth "
              "to demonstrate the reporting mechanism. Real results will replace "
              "this automatically once genuine 90-day-old predictions exist.")
        log_df = pd.read_csv(LOG_PATH)
        eligible = log_df.copy()
        is_simulated = True

    if eligible.empty:
        print("No predictions available at all. Run the API/Gradio/batch scoring first.")
        return

    eligible = simulate_ground_truth(eligible) if is_simulated else eligible

    if "actual_churn" not in eligible.columns:
        print("No ground truth column available — cannot compute realized metrics.")
        return

    y_true = eligible["actual_churn"]
    y_pred = eligible["churn_prediction"]

    metrics = {
        "is_simulated": is_simulated,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_predictions_evaluated": len(eligible),
        "realized_precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "realized_recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "realized_f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    os.makedirs("artifacts", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_feedback_loop()