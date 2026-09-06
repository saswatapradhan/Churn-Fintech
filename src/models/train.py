"""Train XGBoost churn model. (Model Training step)"""
"""
Model Training — PayPal SMB EU Churn
Trains XGBoost on the feature-engineered dataset, evaluates, and saves:
  - the trained model (src/serving/model/model.xgb)
  - metrics.json (this run's results)
  - model_baseline.json (ONLY if it doesn't exist yet — first run becomes baseline)
"""
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

FEATURES_PATH = "data/processed/paypal_smb_eu_churn_features.csv"
MODEL_PATH = "src/serving/model/model.xgb"
METRICS_PATH = "artifacts/metrics.json"
BASELINE_PATH = "config/model_baseline.json"
ID_COL = "account_id"
TARGET_COL = "churn"

def load_features():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def train_and_evaluate():
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    metrics = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }

    print(classification_report(y_test, y_pred, digits=3))
    print("Metrics:", metrics)

    return model, metrics

if __name__ == "__main__":
    model, metrics = train_and_evaluate()

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("src/serving/model", exist_ok=True)

    model.save_model(MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")

    if not os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"No baseline existed — this run IS the new baseline: {BASELINE_PATH}")
    else:
        print(f"Baseline already exists at {BASELINE_PATH} — not overwritten. "
              f"(model_quality_gate.py will compare against it next.)")