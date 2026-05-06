#!/bin/bash
for lr in 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2; do
    python slurm/submit_leonardo.py --time 00:30:00 --job-name "lr_${lr}" -- \
        python scripts/train.py --config-dir "experiments/04_lr_comparison/lr_${lr}" experiment=config
done
