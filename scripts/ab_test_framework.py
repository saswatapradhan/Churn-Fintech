"""
A/B Testing Framework — PayPal SMB EU Churn Retention Outreach
Tests whether Sales outreach to flagged high-risk accounts actually reduces
churn, vs. a control group that receives no intervention. Uses a two-proportion
z-test for statistical significance — the same rigor used in real experiment
platforms (Optimizely, in-house A/B tools).

Real usage: run assign_experiment_groups() once when accounts are flagged.
After the observation window (e.g. 90 days), run analyze_experiment() against
real outcome data.
"""
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime, timezone
from statsmodels.stats.proportion import proportions_ztest

SCORED_ACCOUNTS_PATH = "data/processed/scored_accounts.csv"
EXPERIMENT_OUTPUT_PATH = "data/processed/ab_test_assignments.csv"
RESULTS_PATH = "artifacts/ab_test_results.json"

TREATMENT_EFFECT_RELATIVE_REDUCTION = 0.15  # SIMULATED: assumed 15% relative churn reduction from outreach
SIGNIFICANCE_LEVEL = 0.05

def assign_group(account_id: str) -> str:
    """
    Deterministic 50/50 split by hashing account_id — same account always
    gets the same group, no randomization drift across re-runs.
    """
    hash_val = int(hashlib.md5(account_id.encode()).hexdigest(), 16)
    return "treatment" if hash_val % 2 == 0 else "control"

def assign_experiment_groups(risk_threshold: str = "HIGH") -> pd.DataFrame:
    """Real logic: assigns HIGH-risk accounts to treatment/control groups."""
    df = pd.read_csv(SCORED_ACCOUNTS_PATH)
    eligible = df[df["risk_level"] == risk_threshold].copy()

    eligible["experiment_group"] = eligible["account_id"].apply(assign_group)
    eligible["assigned_at"] = datetime.now(timezone.utc).isoformat()

    eligible.to_csv(EXPERIMENT_OUTPUT_PATH, index=False)
    return eligible

def simulate_outcomes(assignments: pd.DataFrame) -> pd.DataFrame:
    """
    SIMULATED — stands in for real 90-day churn outcomes. Treatment group's
    churn probability is reduced by TREATMENT_EFFECT_RELATIVE_REDUCTION to
    demonstrate the analysis working; replace with real outcome data once
    outreach has actually run and enough time has passed.
    """
    np.random.seed(42)
    assignments = assignments.copy()

    base_prob = assignments["churn_probability"]
    effective_prob = np.where(
        assignments["experiment_group"] == "treatment",
        base_prob * (1 - TREATMENT_EFFECT_RELATIVE_REDUCTION),
        base_prob
    )
    assignments["actual_churned"] = (np.random.uniform(0, 1, len(assignments)) < effective_prob).astype(int)
    return assignments

def analyze_experiment(assignments: pd.DataFrame, is_simulated: bool) -> dict:
    treatment = assignments[assignments["experiment_group"] == "treatment"]
    control = assignments[assignments["experiment_group"] == "control"]

    treatment_churn_rate = treatment["actual_churned"].mean()
    control_churn_rate = control["actual_churned"].mean()

    count = np.array([treatment["actual_churned"].sum(), control["actual_churned"].sum()])
    nobs = np.array([len(treatment), len(control)])

    z_stat, p_value = proportions_ztest(count, nobs)

    relative_reduction = (control_churn_rate - treatment_churn_rate) / control_churn_rate if control_churn_rate > 0 else 0

    return {
        "is_simulated": is_simulated,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "treatment_group_size": len(treatment),
        "control_group_size": len(control),
        "treatment_churn_rate": round(float(treatment_churn_rate), 4),
        "control_churn_rate": round(float(control_churn_rate), 4),
        "relative_churn_reduction": round(float(relative_reduction), 4),
        "z_statistic": round(float(z_stat), 4),
        "p_value": round(float(p_value), 4),
        "statistically_significant": bool(p_value < SIGNIFICANCE_LEVEL),
        "conclusion": (
            "Outreach shows a statistically significant reduction in churn."
            if p_value < SIGNIFICANCE_LEVEL and treatment_churn_rate < control_churn_rate
            else "No statistically significant effect detected — insufficient evidence "
                 "that outreach changes churn outcomes at current sample size."
        )
    }

if __name__ == "__main__":
    print("Assigning experiment groups...")
    assignments = assign_experiment_groups()
    print(f"Assigned {len(assignments)} HIGH-risk accounts: "
          f"{(assignments['experiment_group'] == 'treatment').sum()} treatment, "
          f"{(assignments['experiment_group'] == 'control').sum()} control")

    print("\nSimulating outcomes (no real outreach/outcome data exists yet)...")
    assignments_with_outcomes = simulate_outcomes(assignments)

    results = analyze_experiment(assignments_with_outcomes, is_simulated=True)

    print(json.dumps(results, indent=2))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")