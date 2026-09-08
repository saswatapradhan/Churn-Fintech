"""
S3 Helper — PayPal SMB EU Churn
Centralizes all S3 read/write logic so every script (batch scoring, drift
checks, feedback loop) uses the same connection pattern.
"""
import boto3
import os
import io
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )

def get_bucket_name():
    return os.environ["S3_BUCKET_NAME"]

def upload_dataframe_to_s3(df: pd.DataFrame, s3_key: str):
    """Uploads a DataFrame as CSV directly to S3, no local file needed."""
    s3 = get_s3_client()
    bucket = get_bucket_name()
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"Uploaded to s3://{bucket}/{s3_key}")

def download_dataframe_from_s3(s3_key: str) -> pd.DataFrame:
    """Downloads a CSV from S3 directly into a DataFrame, no local file needed."""
    s3 = get_s3_client()
    bucket = get_bucket_name()
    response = s3.get_object(Bucket=bucket, Key=s3_key)
    return pd.read_csv(io.BytesIO(response["Body"].read()))

def upload_json_to_s3(data: dict, s3_key: str):
    """Uploads a dict as JSON directly to S3."""
    import json
    s3 = get_s3_client()
    bucket = get_bucket_name()
    s3.put_object(Bucket=bucket, Key=s3_key, Body=json.dumps(data, indent=2))
    print(f"Uploaded to s3://{bucket}/{s3_key}")

def download_json_from_s3(s3_key: str) -> dict:
    """Downloads a JSON object from S3 as a dict."""
    import json
    s3 = get_s3_client()
    bucket = get_bucket_name()
    response = s3.get_object(Bucket=bucket, Key=s3_key)
    return json.loads(response["Body"].read().decode("utf-8"))

def file_exists_in_s3(s3_key: str) -> bool:
    s3 = get_s3_client()
    bucket = get_bucket_name()
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        return True
    except s3.exceptions.ClientError:
        return False