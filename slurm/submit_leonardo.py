#!/usr/bin/env python3
"""Submit an sbatch job on Leonardo with parameterized GPU count."""

import argparse
import subprocess
import sys

CPUS_PER_TASK = 8
TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={gpus}
#SBATCH --gpus-per-node={gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

module load python/3.11.7
cd /leonardo/home/userexternal/jschrei1/equicast
source venv/bin/activate

mkdir -p logs

srun {train_cmd}
"""


def main():
    parser = argparse.ArgumentParser(description="Submit a Leonardo sbatch job.")
    parser.add_argument("command", help="Command to run, as a quoted string (e.g. 'python config/train.py --backbone painn')")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1). Sets ntasks=N, cpus=N*8.")
    parser.add_argument("--time", default="12:00:00", help="Wall time (default: 12:00:00)")
    parser.add_argument("--job-name", default="equicast", help="Job name (default: equicast)")
    parser.add_argument("--partition", default="boost_usr_prod")
    parser.add_argument("--account", default="deste_340_26")
    parser.add_argument("--dry-run", action="store_true", help="Print the script instead of submitting")
    args = parser.parse_args()

    train_cmd = args.command

    script = TEMPLATE.format(
        job_name=args.job_name,
        partition=args.partition,
        account=args.account,
        gpus=args.gpus,
        cpus_per_task=CPUS_PER_TASK,
        time=args.time,
        train_cmd=train_cmd,
    )

    if args.dry_run:
        print(script)
        return

    result = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
