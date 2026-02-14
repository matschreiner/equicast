#!/bin/bash
#SBATCH --job-name=equicast-test
#SBATCH --partition=boost_usr_prod
#SBATCH --account=deste_340_26
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

module load python/3.11.7
cd /leonardo/home/userexternal/jschrei1/equicast
source venv/bin/activate

mkdir -p logs

srun python config/train.py --backbone painn --fiddler leonardo
