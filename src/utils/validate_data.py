"""
Data Validation — PayPal SMB EU Churn Dataset
Mirrors the Telco project's Great Expectations validation step.
Runs BEFORE any preprocessing/training — same as the CI/CD pipeline gate.
"""
import pandas as pd
import great_expectations as gx
import json

df = pd.read_csv('/mnt/user-data/outputs/paypal_smb_eu_churn.csv')

context = gx.get_context()
data_source = context.data_sources.add_pandas("pandas_source")
data_asset = data_source.add_dataframe_asset(name="paypal_smb_asset")
batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

suite = context.suites.add(gx.ExpectationSuite(name="paypal_smb_eu_churn_suite"))

expectations = [
    # Identity / schema integrity
    gx.expectations.ExpectColumnValuesToBeUnique(column="account_id"),
    gx.expectations.ExpectColumnValuesToNotBeNull(column="account_id"),

    # Categorical domain checks
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="country", value_set=["DE", "FR", "NL", "IT", "ES", "UK"]),
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="industry", value_set=["Retail", "E-commerce", "Hospitality", "Services", "Subscription"]),

    # Target sanity
    gx.expectations.ExpectColumnValuesToBeInSet(column="churn", value_set=[0, 1]),
    gx.expectations.ExpectColumnValuesToNotBeNull(column="churn"),

    # Numeric range sanity (business logic, not just "not null")
    gx.expectations.ExpectColumnValuesToBeBetween(column="account_age_months", min_value=1, max_value=200),
    gx.expectations.ExpectColumnValuesToBeBetween(column="monthly_tpv", min_value=0, max_value=None),
    gx.expectations.ExpectColumnValuesToBeBetween(column="tpv_trend_3m_pct", min_value=-100, max_value=500),
    gx.expectations.ExpectColumnValuesToBeBetween(column="txn_count_monthly", min_value=0, max_value=None),
    gx.expectations.ExpectColumnValuesToBeBetween(column="avg_ticket_size", min_value=0, max_value=None),
    gx.expectations.ExpectColumnValuesToBeBetween(column="num_products_used", min_value=1, max_value=4),
    gx.expectations.ExpectColumnValuesToBeBetween(column="login_freq_monthly", min_value=0, max_value=None),
    gx.expectations.ExpectColumnValuesToBeBetween(column="disputes_90d", min_value=0, max_value=None),
    gx.expectations.ExpectColumnValuesToBeBetween(column="chargebacks_90d", min_value=0, max_value=None),
    gx.expectations.ExpectColumnValuesToBeBetween(column="support_tickets_90d", min_value=0, max_value=None),
    gx.expectations.ExpectColumnValuesToBeBetween(column="failed_txn_ratio", min_value=0, max_value=1),

    # No column should be entirely null
    gx.expectations.ExpectTableRowCountToBeBetween(min_value=1000, max_value=None),
]

for exp in expectations:
    suite.add_expectation(exp)

results = batch.validate(suite)

# Summarize like a CI/CD gate would
summary = {
    "success": results.success,
    "total_expectations": len(results.results),
    "successful_expectations": sum(1 for r in results.results if r.success),
    "failed_expectations": [
        r.expectation_config.type for r in results.results if not r.success
    ]
}

print(json.dumps(summary, indent=2))

with open('/mnt/user-data/outputs/data_validation_report.json', 'w') as f:
    json.dump(summary, f, indent=2)

if not summary["success"]:
    raise SystemExit("❌ DATA VALIDATION FAILED — pipeline should stop here (CI/CD gate).")
else:
    print("✅ DATA VALIDATION PASSED — safe to proceed to preprocessing/training.")
