"""Optuna hyperparameter tuning. (Model Tuning step)"""
"""
Hyperparameter Tuning — PayPal SMB EU Churn
Runs Optuna to maximize recall (catching churners matters most for early warning),
then retrains XGBoost with the best params and saves it as the candidate model.
"""
import pandas as pd
import json
import os
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

FEATURES_PATH = "data/processed/paypal_smb_eu_churn_features.csv"
MODEL_PATH = "src/serving/model/model.xgb"
METRICS_PATH = "artifacts/metrics.json"
ID_COL = "account_id"
TARGET_COL = "churn"
N_TRIALS = 25
THRESHOLD = 0.5

def load_split():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    return recall_score(y_test, y_pred)

def tune_and_train():
    X_train, X_test, y_train, y_test = load_split()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X_train, X_test, y_train, y_test), n_trials=N_TRIALS)

    print("Best params:", study.best_params)
    print("Best recall (during search):", study.best_value)

    best_params = study.best_params
    best_params.update({
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    })

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    metrics = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }

    print(classification_report(y_test, y_pred, digits=3))
    print("Final tuned metrics:", metrics)
    return final_model, metrics, best_params

if __name__ == "__main__":
    model, metrics, best_params = tune_and_train()

    os.makedirs("artifacts", exist_ok=True)
    model.save_model(MODEL_PATH)  # overwrites baseline model with tuned candidate

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    with open("artifacts/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Saved tuned model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved best params: artifacts/best_params.json")