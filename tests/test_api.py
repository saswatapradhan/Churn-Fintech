"""Tests for FastAPI /predict endpoint."""
"""
Tests for the FastAPI /predict endpoint.
Run: pytest tests/test_api.py -v
"""
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "account_id": "TEST_API_001",
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

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint_success():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == "TEST_API_001"
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= body["churn_probability"] <= 1.0

def test_predict_endpoint_missing_field_returns_422():
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["country"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # FastAPI/Pydantic validation error

def test_predict_endpoint_wrong_type_returns_422():
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["monthly_tpv"] = "not_a_number"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422