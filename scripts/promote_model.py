"""
Model Promotion — PayPal SMB EU Churn
Promotes the latest registered model version from Staging to Production
in the MLflow Model Registry. Run ONLY after model_quality_gate.py passes.
Connects to MLflow backed by a real Postgres RDS instance (not local SQLite).
"""
import os
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "paypal-smb-churn"

def get_mlflow_tracking_uri() -> str:
    return (
        f"postgresql://{os.environ['MLFLOW_DB_USER']}:{os.environ['MLFLOW_DB_PASSWORD']}"
        f"@{os.environ['MLFLOW_DB_HOST']}:{os.environ['MLFLOW_DB_PORT']}/{os.environ['MLFLOW_DB_NAME']}"
    )

def promote_latest_to_production():
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise ValueError(f"No versions found for model '{MODEL_NAME}'")

    latest_version = max(versions, key=lambda v: int(v.version))

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True  # old Production version becomes Archived, not deleted
    )

    print(f"Promoted {MODEL_NAME} version {latest_version.version} to Production")
    print(f"(Previous Production version, if any, archived — not deleted, so rollback is possible)")

if __name__ == "__main__":
    promote_latest_to_production()