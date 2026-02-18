#!/bin/bash
# Sync mlflow db and job logs from Leonardo.
REMOTE="leonardo:/leonardo/home/userexternal/jschrei1/equicast"

# Checkpoint WAL into main db before syncing to avoid corruption.
ssh leonardo "cd /leonardo/home/userexternal/jschrei1/equicast && sqlite3 mlflow/mlflow.db 'PRAGMA wal_checkpoint(FULL);'" 2>/dev/null

rsync -avpr "$REMOTE/mlflow/" mlflow/
rsync -avpr "$REMOTE/jobs/" jobs/
