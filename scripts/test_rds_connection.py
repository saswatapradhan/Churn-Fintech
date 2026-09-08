"""Quick test: confirm we can connect to the RDS Postgres instance."""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.environ["MLFLOW_DB_HOST"],
    port=os.environ["MLFLOW_DB_PORT"],
    dbname=os.environ["MLFLOW_DB_NAME"],
    user=os.environ["MLFLOW_DB_USER"],
    password=os.environ["MLFLOW_DB_PASSWORD"],
)

cursor = conn.cursor()
cursor.execute("SELECT version();")
print("Connected successfully!")
print(cursor.fetchone())

cursor.close()
conn.close()