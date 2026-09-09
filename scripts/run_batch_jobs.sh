#!/bin/bash
set -e
python -m scripts.batch_score
python -m scripts.check_drift
echo "Batch jobs completed successfully"