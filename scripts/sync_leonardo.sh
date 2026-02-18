#!/bin/bash
# Sync mlflow db and job logs from Leonardo.
REMOTE="leonardo:/leonardo/home/userexternal/jschrei1/equicast"

rsync -avpr "$REMOTE/mlflow/" mlflow/
rsync -avpr "$REMOTE/jobs/" jobs/
