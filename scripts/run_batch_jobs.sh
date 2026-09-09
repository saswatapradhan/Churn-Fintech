#!/bin/bash
set -e
python -m scripts.batch_score
python scripts/check_drift.py
echo "Batch jobs completed successfully"