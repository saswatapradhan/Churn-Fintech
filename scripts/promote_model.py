"""
Model Promotion — PayPal SMB EU Churn
Promotes the latest registered model version from Staging to Production
in the MLflow Model Registry. Run ONLY after model_quality_gate.py passes.
"""
import mlflow
from mlflow import MlflowClient

MODEL_NAME = "paypal-smb-churn"

def promote_latest_to_production():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
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