"""Compares new model metrics against config/model_baseline.json; fails CI if regressed."""
"""
Model Quality Gate — PayPal SMB EU Churn
Compares the latest trained model's metrics against the stored baseline.
Fails (non-zero exit) if the candidate model regresses on recall — our
primary business metric for an early-warning churn system.
Run this after train.py or tune.py, before deploying/serving the model.
"""
import json
import sys

METRICS_PATH = "artifacts/metrics.json"
BASELINE_PATH = "config/model_baseline.json"

# How much regression we tolerate before failing the gate.
# Recall matters most for our business case (early warning), so it's the strictest.
TOLERANCE = {
    "recall": -0.03,     # allow at most a 3-point recall drop
    "precision": -0.05,  # allow at most a 5-point precision drop
    "roc_auc": -0.02,    # allow at most a 2-point ROC-AUC drop
}

def load_json(path):
    with open(path) as f:
        return json.load(f)

def run_gate():
    metrics = load_json(METRICS_PATH)
    baseline = load_json(BASELINE_PATH)

    print("Baseline:", baseline)
    print("Candidate:", metrics)

    failures = []
    for metric, min_delta in TOLERANCE.items():
        delta = metrics[metric] - baseline[metric]
        status = "OK" if delta >= min_delta else "FAIL"
        print(f"  {metric}: baseline={baseline[metric]:.4f} candidate={metrics[metric]:.4f} "
              f"delta={delta:+.4f} [{status}]")
        if delta < min_delta:
            failures.append(metric)

    if failures:
        print(f"\n❌ QUALITY GATE FAILED — regression in: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("\n✅ QUALITY GATE PASSED — candidate model is acceptable to deploy.")

if __name__ == "__main__":
    run_gate()