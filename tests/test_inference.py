"""Tests for model inference logic."""
"""
Tests for the core inference logic — independent of any web framework.
Run: pytest tests/test_inference.py -v
"""
import pytest
from src.serving.inference import predict_churn, add_derived_features
import pandas as pd

HIGH_RISK_ACCOUNT = {
    "account_id": "TEST_HIGH_RISK",
    "country": "DE",
    "industry": "E-commerce",
    "account_age_months": 8,
    "monthly_tpv": 4500,
    "tpv_trend_3m_pct": -45.0,
    "txn_count_monthly": 30,
    "avg_ticket_size": 150,
    "num_products_used": 1,
    "login_freq_monthly": 2,
    "disputes_90d": 4,
    "chargebacks_90d": 2,
    "support_tickets_90d": 3,
    "failed_txn_ratio": 0.25
}

HEALTHY_ACCOUNT = {
    "account_id": "TEST_HEALTHY",
    "country": "FR",
    "industry": "Retail",
    "account_age_months": 40,
    "monthly_tpv": 12000,
    "tpv_trend_3m_pct": 15.0,
    "txn_count_monthly": 80,
    "avg_ticket_size": 150,
    "num_products_used": 4,
    "login_freq_monthly": 20,
    "disputes_90d": 0,
    "chargebacks_90d": 0,
    "support_tickets_90d": 0,
    "failed_txn_ratio": 0.02
}

def test_predict_returns_expected_keys():
    result = predict_churn(HIGH_RISK_ACCOUNT)
    assert set(result.keys()) == {"churn_prediction", "churn_probability", "risk_level"}

def test_high_risk_account_flagged_high():
    result = predict_churn(HIGH_RISK_ACCOUNT)
    assert result["risk_level"] == "HIGH"
    assert result["churn_probability"] > 0.5

def test_healthy_account_flagged_low_or_medium():
    result = predict_churn(HEALTHY_ACCOUNT)
    assert result["risk_level"] in ("LOW", "MEDIUM")
    assert result["churn_probability"] < 0.5

def test_probability_is_valid_range():
    result = predict_churn(HIGH_RISK_ACCOUNT)
    assert 0.0 <= result["churn_probability"] <= 1.0

def test_prediction_is_binary():
    result = predict_churn(HIGH_RISK_ACCOUNT)
    assert result["churn_prediction"] in (0, 1)

def test_add_derived_features_handles_string_inputs():
    """Regression test for the Gradio string-type bug we just fixed."""
    df = pd.DataFrame([{
        "disputes_90d": "4", "chargebacks_90d": "2", "txn_count_monthly": "30",
        "login_freq_monthly": "2", "num_products_used": "1", "tpv_trend_3m_pct": "-45.0"
    }])
    result = add_derived_features(df)
    assert result["dispute_to_txn_ratio"].iloc[0] == pytest.approx(4 / 30)
    assert result["is_sharp_decline"].iloc[0] == 1

def test_add_derived_features_handles_zero_transactions():
    """Guards against division-by-zero for dormant/new accounts."""
    df = pd.DataFrame([{
        "disputes_90d": 2, "chargebacks_90d": 1, "txn_count_monthly": 0,
        "login_freq_monthly": 0, "num_products_used": 1, "tpv_trend_3m_pct": -10.0
    }])
    result = add_derived_features(df)
    assert not result["dispute_to_txn_ratio"].isna().any()
    assert not result["dispute_to_txn_ratio"].isin([float("inf")]).any()