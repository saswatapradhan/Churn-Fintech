"""FastAPI app exposing /predict for PayPal SMB EU Churn model. Hardened with API key auth + rate limiting."""
import os
from fastapi import FastAPI, Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.serving.inference import predict_churn

# ---------------- API Key Setup ----------------
# In production, this comes from a secrets manager (AWS Secrets Manager, etc.),
# not an env var default — the fallback here is ONLY for local dev convenience.
API_KEY = os.environ.get("CHURN_API_KEY", "dev-local-key-change-me")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# ---------------- Rate Limiter Setup ----------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="PayPal SMB EU Churn Prediction API", version="1.1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("30/minute")  # generous for a Sales tool, tight enough to stop abuse
def predict(request: Request, account: SMBAccount, api_key: str = Depends(verify_api_key)):
    result = predict_churn(account.model_dump(), source="api")
    return {"account_id": account.account_id, **result}