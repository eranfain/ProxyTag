#!/usr/bin/env python3
"""
Run DCNv2 and AutoInt with all tag variants per dataset, then produce a comparison table.

For each dataset, runs:
  - No-tags baseline (DCNv2, AutoInt)
  - Each tag variant file found in data/tags/{dataset}/full/

Results are saved to {results_dir}/{dataset}_comparison.csv
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "amazon-books": {
        "train_path": "data/amazon-books/interactions/train.parquet",
        "val_path": "data/amazon-books/interactions/val.parquet",
        "test_path": "data/amazon-books/interactions/test.parquet",
        "user_id_col": "userId",
        "item_id_col": "asin",
        "tags_dir": "data/tags/amazon-books/full",
    },
    "amazon-movies": {
        "train_path": "data/amazon-movies/interactions/train.parquet",
        "val_path": "data/amazon-movies/interactions/val.parquet",
        "test_path": "data/amazon-movies/interactions/test.parquet",
        "user_id_col": "userId",
        "item_id_col": "asin",
        "tags_dir": "data/tags/amazon-movies/full",
    },
}

MODELS = ["dcnv2", "autoint", "unisrec", "kar", "llmrec"]

# Per-dataset, per-model hyperparameter overrides.
# These address overfitting on smaller/denser datasets where models converge too fast
# and ignore tag features. Values tuned to keep train loss in a regime where tags
# still modulate the loss (i.e., not saturated near zero).
HYPERPARAMS = {
    "amazon-books": {
        "dcnv2": {},
        "autoint": {},
        "kar": {},
        "llmrec": {},
        "unisrec": {"--batch_size": "512"},
    },
    "amazon-movies": {
        "dcnv2": {},
        "autoint": {},
        "kar": {},
        "llmrec": {},
        "unisrec": {"--batch_size": "512"},
    },
}

MODULE_MAP = {}


def build_jobs(dataset_name, models, results_dir, max_tags=16):
    """Build list of (job_name, cmd) for a dataset. Skips jobs with existing results."""
    cfg = DATASETS[dataset_name]
    tag_files = sorted(glob.glob(os.path.join(cfg["tags_dir"], "*.parquet")))

    jobs = []
    skipped = 0
    for model in models:
        module = MODULE_MAP.get(model, f"proxytag.baselines.{model}")
        base_args = [
            sys.executable, "-m", module,
            "--train_path", cfg["train_path"],
            "--val_path", cfg["val_path"],
            "--test_path", cfg["test_path"],
            "--user_id_col", cfg["user_id_col"],
            "--item_id_col", cfg["item_id_col"],
            "--results_dir", results_dir,
        ]

        # UniSRec needs --train_stage
        if model == "unisrec":
            base_args += ["--train_stage", "transductive_ft"]

        # Apply per-dataset, per-model hyperparameter overrides
        hp_overrides = HYPERPARAMS.get(dataset_name, {}).get(model, {})
        hp_args = []
        for flag, value in hp_overrides.items():
            hp_args += [flag, value]

        # No-tags baseline (skip for models that require tags)
        tags_required_models = {"unisrec", "kar", "llmrec"}
        if model not in tags_required_models:
            job_name = f"{model}_no_tags_{dataset_name}"
            results_file = os.path.join(results_dir, f"{job_name}_results.json")
            if os.path.exists(results_file):
                skipped += 1
            else:
                cmd = base_args + hp_args + ["--job_name", job_name]
                jobs.append((job_name, cmd))

        # Each tag variant
        max_tags_flag = "--max_tags"
        for tag_file in tag_files:
            variant = os.path.splitext(os.path.basename(tag_file))[0]
            job_name = f"{model}_{variant}_{dataset_name}"
            results_file = os.path.join(results_dir, f"{job_name}_results.json")
            if os.path.exists(results_file):
                skipped += 1
            else:
                cmd = base_args + hp_args + [
                    "--tags_path", tag_file,
                    max_tags_flag, str(max_tags),
                    "--job_name", job_name,
                ]
                jobs.append((job_name, cmd))

    if skipped:
        print(f"  Skipped {skipped} jobs with existing results")

    return jobs


def run_jobs(jobs, max_parallel=1):
    """Run jobs sequentially or in parallel."""
    succeeded = []
    failed = []

    if max_parallel == 1:
        for job_name, cmd in jobs:
            print(f"\n[RUNNING] {job_name}")
            result = subprocess.run(cmd)
            if result.returncode == 0:
                succeeded.append(job_name)
                print(f"[DONE] {job_name}")
            else:
                failed.append(job_name)
                print(f"[FAILED] {job_name} (code={result.returncode})")
    else:
        running = {}
        pending = list(jobs)
        import time

        try:
            while pending or running:
                while pending and len(running) < max_parallel:
                    job_name, cmd = pending.pop(0)
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    running[proc.pid] = (proc, job_name)
                    print(f"[STARTED] {job_name} (pid={proc.pid})")

                finished_pids = []
                for pid, (proc, job_name) in running.items():
                    ret = proc.poll()
                    if ret is not None:
                        finished_pids.append(pid)
                        if ret == 0:
                            succeeded.append(job_name)
                            print(f"[DONE] {job_name}")
                        else:
                            failed.append(job_name)
                            output = proc.stdout.read()
                            print(f"[FAILED] {job_name} (code={ret})")
                            if output.strip():
                                last_lines = "\n".join(output.strip().split("\n")[-3:])
                                print(f"  {last_lines}")

                for pid in finished_pids:
                    del running[pid]

                if running and not finished_pids:
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\nInterrupted! Terminating...")
            for pid, (proc, _) in running.items():
                proc.terminate()
            for pid, (proc, _) in running.items():
                proc.wait()
            sys.exit(1)

    return succeeded, failed


def build_comparison_table(results_dir, dataset_name):
    """Load all result JSONs for a dataset and build a comparison DataFrame."""
    pattern = os.path.join(results_dir, f"*_{dataset_name}_results.json")
    result_files = sorted(glob.glob(pattern))

    rows = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)

        job_name = data.get("job_name", "")
        metrics = data.get("test_metrics", {})

        # Parse model and variant from job_name
        # Format: {model}_{variant}_{dataset} or {model}_no_tags_{dataset}
        parts = job_name.replace(f"_{dataset_name}", "")
        model = data.get("model", "unknown")
        prefix_map = {
            "dcnv2_": "DCNv2",
            "autoint_": "AutoInt",
            "kar_": "KAR",
            "llmrec_": "LLM-Rec",
            "unisrec_": "UniSRec",
        }
        variant = parts
        for prefix, name in prefix_map.items():
            if parts.startswith(prefix):
                model = name
                variant = parts[len(prefix):]
                break

        rows.append({
            "model": model,
            "tag_variant": variant if variant != "no_tags" else "(no tags)",
            "ndcg@10": metrics.get("ndcg@10", 0.0),
            "recall@10": metrics.get("recall@10", 0.0),
            "n_eval": metrics.get("n_eval", 0),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(f"No results found matching: {pattern}")
        return None

    df = df.sort_values(["model", "ndcg@10"], ascending=[True, False])
    return df


def main():
    parser = argparse.ArgumentParser(description="Run all baseline experiments and produce comparison table")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                        default=list(DATASETS.keys()),
                        help="Datasets to run (default: all)")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                        help="Models to run (default: dcnv2 autoint)")
    parser.add_argument("--max_tags", type=int, default=16,
                        help="Max tags per item (default: 16)")
    parser.add_argument("--max_parallel", type=int, default=1,
                        help="Max parallel jobs (default: 1, sequential)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for result JSONs and comparison tables")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--table_only", action="store_true",
                        help="Skip training, just build table from existing results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    for dataset_name in args.datasets:
        print("\n" + "=" * 80)
        print(f"DATASET: {dataset_name}")
        print("=" * 80)

        if not args.table_only:
            jobs = build_jobs(dataset_name, args.models, args.results_dir, max_tags=args.max_tags)

            if args.dry_run:
                print(f"\nDry run: {len(jobs)} jobs")
                for job_name, cmd in jobs:
                    print(f"  {job_name}")
                    print(f"    {' '.join(cmd)}\n")
                continue

            print(f"\nRunning {len(jobs)} jobs...")
            succeeded, failed = run_jobs(jobs, max_parallel=args.max_parallel)

            print(f"\n  Succeeded: {len(succeeded)}, Failed: {len(failed)}")
            if failed:
                print("  Failed jobs:")
                for j in failed:
                    print(f"    - {j}")

        # Build comparison table
        print(f"\n{'=' * 80}")
        print(f"COMPARISON TABLE: {dataset_name}")
        print(f"{'=' * 80}")

        table = build_comparison_table(args.results_dir, dataset_name)
        if table is not None:
            print(table.to_string(index=False))

            # Save to CSV
            csv_path = os.path.join(args.results_dir, f"{dataset_name}_comparison.csv")
            table.to_csv(csv_path, index=False)
            print(f"\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()
