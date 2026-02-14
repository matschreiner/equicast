#!/usr/bin/env python3
"""Submit an sbatch job on Leonardo with parameterized GPU count.

Usage:
    python slurm/submit_leonardo.py [options] -- command [args...]

Examples:
    # Submit a training job (1 GPU, 12h default)
    python slurm/submit_leonardo.py -- python config/train.py --backbone painn

    # Multi-GPU with custom wall time
    python slurm/submit_leonardo.py --gpus 4 --time 24:00:00 -- python config/train.py --backbone painn

    # Dry run (print script, don't submit)
    python slurm/submit_leonardo.py --dry-run -- python config/train.py --backbone painn

Each submission creates a jobs/<idx>/ folder containing:
    submit.sbatch  - the generated sbatch script
    enter.sh       - executable script to attach to the running job
    slurm-*.out/err - stdout/stderr logs (created by slurm)

To attach to a running job:
    ./jobs/3/enter.sh
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys

CPUS_PER_TASK = 8
PROJECT_DIR = "/leonardo/home/userexternal/jschrei1/equicast"
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
#SBATCH --output={job_dir}/slurm-%j.out
#SBATCH --error={job_dir}/slurm-%j.err

module load python/3.11.7
cd {project_dir}
source venv/bin/activate

srun {train_cmd}
"""

JOBS_DIR = Path(__file__).resolve().parent.parent / "jobs"


def next_job_index() -> int:
    if not JOBS_DIR.exists():
        return 1
    indices = [int(p.name) for p in JOBS_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(indices, default=0) + 1


def main():
    # Split sys.argv on "--" so the command after it isn't parsed by argparse
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        our_args, cmd_args = argv[:split], argv[split + 1 :]
    else:
        our_args, cmd_args = argv, []

    parser = argparse.ArgumentParser(
        description="Submit a Leonardo sbatch job.",
        usage="%(prog)s [options] -- command [args...]",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1). Sets ntasks=N, cpus=N*8.")
    parser.add_argument("--time", default="12:00:00", help="Wall time (default: 12:00:00)")
    parser.add_argument("--job-name", default="equicast", help="Job name (default: equicast)")
    parser.add_argument("--partition", default="boost_usr_prod")
    parser.add_argument("--account", default="deste_340_26")
    parser.add_argument("--dry-run", action="store_true", help="Print the script instead of submitting")
    args = parser.parse_args(our_args)

    if not cmd_args:
        parser.error("No command given. Usage: submit_leonardo.py [options] -- command [args...]")

    idx = next_job_index()
    job_dir = JOBS_DIR / str(idx)
    job_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = " ".join(cmd_args)
    job_dir_remote = f"{PROJECT_DIR}/jobs/{idx}"

    script = TEMPLATE.format(
        job_name=args.job_name,
        partition=args.partition,
        account=args.account,
        gpus=args.gpus,
        cpus_per_task=CPUS_PER_TASK,
        time=args.time,
        job_dir=job_dir_remote,
        project_dir=PROJECT_DIR,
        train_cmd=train_cmd,
    )

    (job_dir / "submit.sbatch").write_text(script)

    if args.dry_run:
        print(script)
        write_enter_script(job_dir, job_id="$1")
        print(f"Job folder: {job_dir}")
        return

    result = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 0:
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        job_id = match.group(1) if match else "$1"
        write_enter_script(job_dir, job_id=job_id)
        print(f"Job folder: {job_dir}")
    sys.exit(result.returncode)


def write_enter_script(job_dir: Path, job_id: str) -> None:
    enter = job_dir / "enter.sh"
    enter.write_text(f'#!/bin/bash\nsrun --overlap --jobid="{job_id}" --pty bash\n')
    enter.chmod(enter.stat().st_mode | 0o111)


if __name__ == "__main__":
    main()
