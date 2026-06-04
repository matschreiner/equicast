"""Generate forecast JSON metadata files for eval zarrs that are missing them.

Usage:
    python tools/generate_eval_json.py <eval_dir> --dataset <dataset_path> --n-input 2
"""

import argparse
import glob
import json
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", help="Directory containing forecast_XXXX.zarr files")
    parser.add_argument("--dataset", required=True, help="Path to the source dataset")
    parser.add_argument("--n-input", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--ckpt-path", default="")
    args = parser.parse_args()

    subdir = os.path.join(args.eval_dir, "forecasts")
    search_dir = subdir if os.path.isdir(subdir) else args.eval_dir
    zarrs = sorted(glob.glob(os.path.join(search_dir, "forecast_*.zarr")))
    created = 0
    for zarr_path in zarrs:
        m = re.search(r"forecast_(\d+)\.zarr$", zarr_path)
        if not m:
            continue
        start_idx = int(m.group(1))
        json_path = zarr_path.replace(".zarr", ".json")
        if os.path.exists(json_path):
            continue
        meta = {
            "model_id": args.model_id,
            "ckpt_path": args.ckpt_path,
            "dataset": args.dataset,
            "start_idx": start_idx,
            "n_steps": args.n_steps,
            "n_input": args.n_input,
            "batch": True,
            "output_dir": args.eval_dir,
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        created += 1

    print(f"Created {created} JSON files in {args.eval_dir}")


if __name__ == "__main__":
    main()
