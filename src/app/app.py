"""Gradio UI for PayPal SMB EU Churn prediction demo — with account lookup."""
import gradio as gr
import pandas as pd
from src.serving.inference import predict_churn

COUNTRIES = ["DE", "FR", "NL", "IT", "ES", "UK"]
INDUSTRIES = ["Retail", "E-commerce", "Hospitality", "Services", "Subscription"]

# Load the account database once at startup (simulates a warehouse/CRM lookup)
ACCOUNTS_DF = pd.read_csv("data/raw/paypal_smb_eu_churn_raw.csv").set_index("account_id")

FIELD_ORDER = [
    "country", "industry", "account_age_months", "monthly_tpv", "tpv_trend_3m_pct",
    "txn_count_monthly", "avg_ticket_size", "num_products_used", "login_freq_monthly",
    "disputes_90d", "chargebacks_90d", "support_tickets_90d", "failed_txn_ratio"
]

def lookup_account(account_id):
    """Auto-fills all fields when a known account_id is entered."""
    account_id = account_id.strip()
    if account_id not in ACCOUNTS_DF.index:
        # Unknown ID — clear fields and warn, rather than silently showing stale data
        return [gr.update(value=None)] * len(FIELD_ORDER) + ["⚠️ Account ID not found in database."]

    row = ACCOUNTS_DF.loc[account_id]
    values = [row[field] for field in FIELD_ORDER]
    return values + [f"✅ Loaded account {account_id}"]

def gradio_predict(account_id, country, industry, account_age_months, monthly_tpv,
                    tpv_trend_3m_pct, txn_count_monthly, avg_ticket_size, num_products_used,
                    login_freq_monthly, disputes_90d, chargebacks_90d, support_tickets_90d,
                    failed_txn_ratio):
    record = {
        "account_id": account_id, "country": country, "industry": industry,
        "account_age_months": int(account_age_months), "monthly_tpv": float(monthly_tpv),
        "tpv_trend_3m_pct": float(tpv_trend_3m_pct), "txn_count_monthly": int(txn_count_monthly),
        "avg_ticket_size": float(avg_ticket_size), "num_products_used": int(num_products_used),
        "login_freq_monthly": int(login_freq_monthly), "disputes_90d": int(disputes_90d),
        "chargebacks_90d": int(chargebacks_90d), "support_tickets_90d": int(support_tickets_90d),
        "failed_txn_ratio": float(failed_txn_ratio)
    }
    result = predict_churn(record)
    return result["risk_level"], f"{result['churn_probability']:.1%}", result["churn_prediction"]

with gr.Blocks(title="PayPal SMB EU Churn — Early Warning Demo") as demo:
    gr.Markdown("## Database Marketing Tool: Enter an Account ID to Pull Live Signals & Score Churn Risk")

    with gr.Row():
        with gr.Column():
            account_id_box = gr.Textbox(label="Account ID", value="SMB_EU_000001",
                                         placeholder="e.g. SMB_EU_000123")
            lookup_status = gr.Textbox(label="Lookup Status", interactive=False)

            country_dd = gr.Dropdown(COUNTRIES, label="Country", interactive=False)
            industry_dd = gr.Dropdown(INDUSTRIES, label="Industry", interactive=False)
            age_box = gr.Number(label="Account Age (months)", interactive=False)
            tpv_box = gr.Number(label="Monthly TPV (€)", interactive=False)
            trend_box = gr.Number(label="TPV Trend 3M (%)", interactive=False)
            txn_box = gr.Number(label="Monthly Transaction Count", interactive=False)
            ticket_box = gr.Number(label="Avg Ticket Size (€)", interactive=False)
            products_box = gr.Number(label="Products Used (1-4)", interactive=False)
            login_box = gr.Number(label="Login Frequency/Month", interactive=False)
            disputes_box = gr.Number(label="Disputes (90d)", interactive=False)
            chargebacks_box = gr.Number(label="Chargebacks (90d)", interactive=False)
            tickets_box = gr.Number(label="Support Tickets (90d)", interactive=False)
            failed_box = gr.Number(label="Failed Transaction Ratio", interactive=False)

            predict_btn = gr.Button("Predict Churn Risk", variant="primary")

        with gr.Column():
            risk_out = gr.Textbox(label="Risk Level")
            proba_out = gr.Textbox(label="Churn Probability")
            pred_out = gr.Number(label="Prediction (1=Churn, 0=Active)")

    all_fields = [country_dd, industry_dd, age_box, tpv_box, trend_box, txn_box,
                  ticket_box, products_box, login_box, disputes_box, chargebacks_box,
                  tickets_box, failed_box]

    # Auto-populate fields the moment the account ID changes (on blur/enter)
    account_id_box.submit(fn=lookup_account, inputs=account_id_box,
                           outputs=all_fields + [lookup_status])
    account_id_box.blur(fn=lookup_account, inputs=account_id_box,
                         outputs=all_fields + [lookup_status])

    predict_btn.click(
        fn=gradio_predict,
        inputs=[account_id_box] + all_fields,
        outputs=[risk_out, proba_out, pred_out]
    )

if __name__ == "__main__":
    demo.launch()