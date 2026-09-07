"""
Automated Retraining Trigger — PayPal SMB EU Churn
Checks drift + realized performance signals. If either crosses threshold,
triggers the full retrain -> quality gate -> promote pipeline automatically.
Refuses to promote a model that fails the quality gate — retraining can run,
but bad models never reach Production without passing the same gate as
any manual run.

Run this on a schedule (e.g. weekly cron / Airflow DAG) in production.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

DRIFT_SUMMARY_PATH = "artifacts/drift_summary.json"
REALIZED_PERFORMANCE_PATH = "artifacts/realized_performance.json"
RETRAIN_LOG_PATH = "artifacts/retrain_decision_log.json"

# Trigger thresholds — deliberately conservative to avoid unnecessary retrains
DRIFT_SHARE_THRESHOLD = 0.5       # retrain if 50%+ of features have drifted
REALIZED_RECALL_FLOOR = 0.60      # retrain if realized recall drops below this

def load_json_safe(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def decide_retrain_needed() -> dict:
    drift = load_json_safe(DRIFT_SUMMARY_PATH)
    realized = load_json_safe(REALIZED_PERFORMANCE_PATH)

    reasons = []

    if drift is not None and drift.get("drifted_column_share", 0) >= DRIFT_SHARE_THRESHOLD:
        reasons.append(f"Drift share {drift['drifted_column_share']:.1%} >= threshold {DRIFT_SHARE_THRESHOLD:.0%}")

    if realized is not None and realized.get("realized_recall", 1.0) < REALIZED_RECALL_FLOOR:
        note = " (simulated data)" if realized.get("is_simulated") else ""
        reasons.append(f"Realized recall {realized['realized_recall']:.1%} < floor {REALIZED_RECALL_FLOOR:.0%}{note}")

    return {
        "retrain_needed": len(reasons) > 0,
        "reasons": reasons,
        "drift_available": drift is not None,
        "realized_performance_available": realized is not None,
    }

def run_step(description: str, command: list) -> bool:
    print(f"\n{'='*60}\n{description}\n{'='*60}")
    result = subprocess.run(command)
    return result.returncode == 0

def execute_retrain_pipeline() -> bool:
    steps = [
        ("Preprocessing", [sys.executable, "-m", "src.data.preprocess"]),
        ("Feature Engineering", [sys.executable, "-m", "src.features.build_features"]),
        ("Baseline Training", [sys.executable, "-m", "src.models.train"]),
        ("Hyperparameter Tuning", [sys.executable, "-m", "src.models.tune"]),
    ]
    for description, command in steps:
        if not run_step(description, command):
            print(f"\n❌ Retrain pipeline stopped: '{description}' failed.")
            return False

    print(f"\n{'='*60}\nModel Quality Gate\n{'='*60}")
    gate_result = subprocess.run([sys.executable, "scripts/model_quality_gate.py"])
    if gate_result.returncode != 0:
        print("\n❌ Quality gate FAILED — retrained model will NOT be promoted.")
        return False

    print(f"\n{'='*60}\nPromoting to Production\n{'='*60}")
    promote_result = subprocess.run([sys.executable, "scripts/promote_model.py"])
    return promote_result.returncode == 0

if __name__ == "__main__":
    decision = decide_retrain_needed()
    decision["checked_at"] = datetime.now(timezone.utc).isoformat()

    print(json.dumps(decision, indent=2))

    if decision["retrain_needed"]:
        print("\n🔄 Retraining triggered.")
        success = execute_retrain_pipeline()
        decision["retrain_executed"] = True
        decision["retrain_succeeded"] = success
    else:
        print("\n✅ No retraining needed — model health within acceptable thresholds.")
        decision["retrain_executed"] = False
        decision["retrain_succeeded"] = None

    os.makedirs("artifacts", exist_ok=True)
    with open(RETRAIN_LOG_PATH, "w") as f:
        json.dump(decision, f, indent=2)