"""
Model Health Dashboard — PayPal SMB EU Churn
For ML/Data teams: monitors prediction volume, prediction distribution shifts,
feature drift, and which model version is live in Production.
Run: streamlit run dashboards/model_health_dashboard.py
"""
import streamlit as st
import pandas as pd
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plotly.express as px
import mlflow
from mlflow import MlflowClient
import os
from dotenv import load_dotenv
load_dotenv()
from src.utils.prediction_logger import read_log

st.set_page_config(page_title="Model Health — PayPal SMB Churn", layout="wide")
st.title("Model Health Dashboard — PayPal SMB EU Churn")

LOG_PATH = "data/logs/predictions_log.csv"
DRIFT_SUMMARY_PATH = "artifacts/drift_summary.json"
DRIFT_REPORT_PATH = "artifacts/drift_report.html"

# ---------------- Section 1: Current Production Model ----------------
st.header("1. Production Model Status")

try:
    db_uri = (
    f"postgresql://{os.environ['MLFLOW_DB_USER']}:{os.environ['MLFLOW_DB_PASSWORD']}"
    f"@{os.environ['MLFLOW_DB_HOST']}:{os.environ['MLFLOW_DB_PORT']}/{os.environ['MLFLOW_DB_NAME']}"
            )
    mlflow.set_tracking_uri(db_uri)
    client = MlflowClient()
    prod_versions = client.get_latest_versions("paypal-smb-churn", stages=["Production"])

    if prod_versions:
        v = prod_versions[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Production Version", v.version)
        col2.metric("Run ID", v.run_id[:8] + "...")
        col3.metric("Status", v.status)
    else:
        st.warning("No model currently in Production stage.")
except Exception as e:
    st.error(f"Could not connect to MLflow registry: {e}")

st.divider()

# ---------------- Section 2: Prediction Volume & Distribution ----------------
st.header("2. Prediction Activity")


log_df = read_log()

if not log_df.empty:
    log_df["logged_at"] = pd.to_datetime(log_df["logged_at"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions Logged", f"{len(log_df):,}")
    col2.metric("Unique Accounts Scored", f"{log_df['account_id'].nunique():,}")
    col3.metric("Avg Churn Probability", f"{log_df['churn_probability'].mean():.1%}")

    if len(log_df) < 20:
        st.info(
            f"Only {len(log_df)} prediction(s) logged so far — volume trends and "
            "distribution shift charts below need more data to be meaningful. "
            "This grows automatically as the API, Gradio app, and batch scoring run over time."
        )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        source_counts = log_df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "count"]
        fig = px.pie(source_counts, names="source", values="count",
                     title="Predictions by Source (API / Gradio / Batch)")
        st.plotly_chart(fig, width='stretch')

    with chart_col2:
        risk_counts = log_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        fig2 = px.pie(risk_counts, names="risk_level", values="count",
                      title="Predictions by Risk Level",
                      color="risk_level",
                      color_discrete_map={"HIGH": "#EF553B", "MEDIUM": "#FFA15A", "LOW": "#00CC96"})
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Recent Predictions")
    st.dataframe(log_df.sort_values("logged_at", ascending=False).head(20), width='stretch')
else:
    st.warning("No prediction log found yet. Run the API, Gradio app, or batch_score.py to generate one.")

st.divider()

# ---------------- Section 3: Data Drift ----------------
st.header("3. Data Drift Status")

from src.utils.s3_helper import download_json_from_s3, file_exists_in_s3

if file_exists_in_s3("drift/latest_summary.json"):
    drift = download_json_from_s3("drift/latest_summary.json")

    
    col1, col2, col3 = st.columns(3)
    col1.metric("Drifted Columns", f"{int(drift['drifted_column_count'])}")
    col2.metric("Drifted Share", f"{drift['drifted_column_share']:.1%}")
    status = "🔴 DRIFT DETECTED" if drift["dataset_drift_detected"] else "🟢 No Significant Drift"
    col3.metric("Status", status)

    scores_df = pd.DataFrame(
        list(drift["per_column_drift_scores"].items()),
        columns=["feature", "drift_score"]
    ).sort_values("drift_score", ascending=False)

    fig3 = px.bar(scores_df, x="feature", y="drift_score",
                  title="Per-Feature Drift Score (threshold: 0.1)")
    fig3.add_hline(y=0.1, line_dash="dash", line_color="red",
                   annotation_text="Drift Threshold")
    st.plotly_chart(fig3, width='stretch')

    st.caption(f"Full interactive report: `{DRIFT_REPORT_PATH}` (open directly in browser)")
else:
    st.warning("No drift report found yet. Run `python scripts/check_drift.py` first.")