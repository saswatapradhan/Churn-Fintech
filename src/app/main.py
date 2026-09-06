"""FastAPI app exposing /predict for PayPal SMB EU Churn model."""
"""FastAPI app exposing /predict for PayPal SMB EU Churn model."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.serving.inference import predict_churn

app = FastAPI(title="PayPal SMB EU Churn Prediction API", version="1.0")

class SMBAccount(BaseModel):
    account_id: str
    country: str = Field(..., description="DE, FR, NL, IT, ES, or UK")
    industry: str = Field(..., description="Retail, E-commerce, Hospitality, Services, or Subscription")
    account_age_months: int
    monthly_tpv: float
    tpv_trend_3m_pct: float
    txn_count_monthly: int
    avg_ticket_size: float
    num_products_used: int
    login_freq_monthly: int
    disputes_90d: int
    chargebacks_90d: int
    support_tickets_90d: int
    failed_txn_ratio: float

@app.get("/")
def root():
    return {"status": "ok", "message": "PayPal SMB EU Churn API is running"}

@app.post("/predict")
def predict(account: SMBAccount):
    result = predict_churn(account.model_dump())
    return {"account_id": account.account_id, **result}