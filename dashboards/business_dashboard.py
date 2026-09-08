"""
Business Dashboard — PayPal SMB EU Churn
For Database Marketing / Sales: view at-risk accounts, prioritized by
probability x revenue, filterable by country/industry/risk level.
Run: streamlit run dashboards/business_dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="PayPal SMB EU Churn — Sales Dashboard", layout="wide")

from src.utils.s3_helper import download_dataframe_from_s3

@st.cache_data(ttl=300)  # refresh from S3 every 5 minutes, not just on restart
def load_data():
    return download_dataframe_from_s3("scored_accounts/latest.csv")


import os
df = load_data()

st.title("PayPal SMB EU Churn — Database Marketing Dashboard")
st.caption(f"Last scored: {df['scored_at'].iloc[0]}")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")
countries = st.sidebar.multiselect("Country", options=sorted(df["country"].unique()), default=None)
industries = st.sidebar.multiselect("Industry", options=sorted(df["industry"].unique()), default=None)
risk_levels = st.sidebar.multiselect("Risk Level", options=["HIGH", "MEDIUM", "LOW"], default=["HIGH"])

filtered = df.copy()
if countries:
    filtered = filtered[filtered["country"].isin(countries)]
if industries:
    filtered = filtered[filtered["industry"].isin(industries)]
if risk_levels:
    filtered = filtered[filtered["risk_level"].isin(risk_levels)]

# ---------------- KPI Row ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accounts (filtered)", f"{len(filtered):,}")
col2.metric("High Risk", f"{(filtered['risk_level'] == 'HIGH').sum():,}")
col3.metric("TPV at Risk (filtered)", f"€{filtered['monthly_tpv'].sum():,.0f}")
col4.metric("Avg Churn Probability", f"{filtered['churn_probability'].mean():.1%}" if len(filtered) else "—")

st.divider()

# ---------------- Charts ----------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    risk_by_country = df.groupby(["country", "risk_level"], observed=True).size().reset_index(name="count")
    fig1 = px.bar(risk_by_country, x="country", y="count", color="risk_level",
                  title="Risk Distribution by Country",
                  color_discrete_map={"HIGH": "#EF553B", "MEDIUM": "#FFA15A", "LOW": "#00CC96"})
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    risk_by_industry = df.groupby(["industry", "risk_level"], observed=True).size().reset_index(name="count")
    fig2 = px.bar(risk_by_industry, x="industry", y="count", color="risk_level",
                  title="Risk Distribution by Industry",
                  color_discrete_map={"HIGH": "#EF553B", "MEDIUM": "#FFA15A", "LOW": "#00CC96"})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------- Priority Account List ----------------
st.subheader("Priority Outreach List (sorted by Probability x TPV)")
st.caption("This is who Sales should call first — not just highest risk, but highest risk x highest revenue.")

display_df = filtered[["account_id", "country", "industry", "monthly_tpv",
                        "churn_probability", "priority_score", "risk_level"]].sort_values(
    "priority_score", ascending=False
)
st.dataframe(
    display_df.style.format({
        "monthly_tpv": "€{:,.0f}",
        "churn_probability": "{:.1%}",
        "priority_score": "{:,.0f}"
    }),
    use_container_width=True,
    height=500
)

st.download_button(
    "Download filtered list as CSV",
    display_df.to_csv(index=False),
    file_name="priority_outreach_list.csv",
    mime="text/csv"
)