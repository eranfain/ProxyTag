#!/usr/bin/env python3
"""
Run HybridRecLightning (main model) with all tag variants per dataset.
Includes a no-tags baseline. Produces a comparison table.

Skips jobs whose results JSON already exists.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

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


def build_jobs(dataset_name, results_dir, sentence_model, max_tags, tag_encoding_type):
    """Build list of (job_name, cmd) for a dataset. Skips existing results."""
    cfg = DATASETS[dataset_name]
    tag_files = sorted(glob.glob(os.path.join(cfg["tags_dir"], "*.parquet")))

    base_args = [
        sys.executable, "scripts/main.py",
        "--train_path", cfg["train_path"],
        "--val_path", cfg["val_path"],
        "--test_path", cfg["test_path"],
        "--user_id_col", cfg["user_id_col"],
        "--item_id_col", cfg["item_id_col"],
        "--sentence_transformers_model", sentence_model,
        "--max_tags", str(max_tags),
        "--tag_encoding_type", tag_encoding_type,
        "--results_dir", results_dir,
    ]

    jobs = []
    skipped = 0

    # No-tags baseline
    job_name = f"hybrid_no_tags_{dataset_name}"
    results_file = os.path.join(results_dir, f"{job_name}_results.json")
    if os.path.exists(results_file):
        skipped += 1
    else:
        cmd = base_args + ["--job_name", job_name, "--no_use_tags"]
        jobs.append((job_name, cmd))

    # Each tag variant
    for tag_file in tag_files:
        variant = os.path.splitext(os.path.basename(tag_file))[0]
        job_name = f"hybrid_{variant}_{dataset_name}"
        results_file = os.path.join(results_dir, f"{job_name}_results.json")
        if os.path.exists(results_file):
            skipped += 1
        else:
            cmd = base_args + [
                "--job_name", job_name,
                "--use_tags",
                "--items_data_path", tag_file,
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
    """Load all hybrid result JSONs for a dataset and build a comparison DataFrame."""
    pattern = os.path.join(results_dir, f"hybrid_*_{dataset_name}_results.json")
    result_files = sorted(glob.glob(pattern))

    rows = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)

        job_name = data.get("job_name", "")
        variant = job_name.replace(f"hybrid_", "").replace(f"_{dataset_name}", "")
        if variant == "no_tags":
            variant = "(no tags)"

        # Handle both metric key formats
        ndcg = data.get("ndcg@10", data.get("test_metrics", {}).get("ndcg@10", 0.0))
        recall = data.get("recall@10", data.get("test_metrics", {}).get("recall@10", 0.0))
        n_eval = data.get("n_eval", data.get("test_metrics", {}).get("n_eval", 0))

        rows.append({
            "tag_variant": variant,
            "ndcg@10": ndcg,
            "recall@10": recall,
            "n_eval": n_eval,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(f"No results found matching: {pattern}")
        return None

    df = df.sort_values("ndcg@10", ascending=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run HybridRec with all tag variants")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                        default=list(DATASETS.keys()),
                        help="Datasets to run (default: all)")
    parser.add_argument("--sentence_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for tag encoding")
    parser.add_argument("--max_tags", type=int, default=16,
                        help="Max tags per item (default: 16)")
    parser.add_argument("--tag_encoding_type", type=str, default="cross_encoder",
                        choices=["cross_encoder", "mean_pooling"],
                        help="Tag encoding type (default: cross_encoder)")
    parser.add_argument("--max_parallel", type=int, default=1,
                        help="Max parallel jobs (default: 1)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for result JSONs")
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
            jobs = build_jobs(
                dataset_name, args.results_dir,
                sentence_model=args.sentence_model,
                max_tags=args.max_tags,
                tag_encoding_type=args.tag_encoding_type,
            )

            if args.dry_run:
                print(f"\nDry run: {len(jobs)} jobs")
                for job_name, cmd in jobs:
                    print(f"  {job_name}")
                    print(f"    {' '.join(cmd)}\n")
                continue

            if not jobs:
                print("  All jobs already completed.")
            else:
                print(f"\nRunning {len(jobs)} jobs...")
                succeeded, failed = run_jobs(jobs, max_parallel=args.max_parallel)
                print(f"\n  Succeeded: {len(succeeded)}, Failed: {len(failed)}")
                if failed:
                    print("  Failed jobs:")
                    for j in failed:
                        print(f"    - {j}")

        # Build comparison table
        print(f"\n{'=' * 80}")
        print(f"COMPARISON TABLE: {dataset_name} (HybridRec)")
        print(f"{'=' * 80}")

        table = build_comparison_table(args.results_dir, dataset_name)
        if table is not None:
            print(table.to_string(index=False))

            csv_path = os.path.join(args.results_dir, f"{dataset_name}_hybrid_comparison.csv")
            table.to_csv(csv_path, index=False)
            print(f"\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()
