#!/usr/bin/env python3
"""
Panel size ablation: run NO@K correlation analysis at different panel sizes
to show stability of the proxy metric.

Produces a table and plot showing how Spearman correlation changes with panel size.
"""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch


PANEL_COUNTS = [50, 100, 200, 500, 1000, 2000]
N_SEEDS = 30


def main():
    parser = argparse.ArgumentParser(description="Panel size ablation for NO@K")
    parser.add_argument('--datasets', nargs='+', default=['amazon-books', 'amazon-movies'])
    parser.add_argument('--panel_counts', nargs='+', type=int, default=PANEL_COUNTS,
                        help=f'Absolute panel item counts to sweep (default: {PANEL_COUNTS})')
    parser.add_argument('--n_seeds', type=int, default=N_SEEDS,
                        help=f'Random seeds per panel size to estimate std (default: {N_SEEDS})')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip panel sizes that already have cached NO@K')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    for dataset_name in args.datasets:
        print("\n" + "=" * 80)
        print(f"PANEL SIZE ABLATION: {dataset_name}")
        print(f"  Panel counts: {args.panel_counts}")
        print(f"  Seeds per count: {args.n_seeds}")
        print("=" * 80)

        # We compute NO@K ourselves (with multiple seeds) instead of calling correlation_analysis.py
        from proxytag.core.data import load_df, build_id_mappings
        from proxytag.analysis.proxy_metric import (
            sample_panel_items, get_cf_item_emb, compute_tag_embeddings, evaluate_tag_quality
        )
        from scripts.correlation_analysis import collect_downstream_metrics
        import glob

        DATASET_CONFIGS = {
            'amazon-books': {
                'train_path': 'data/amazon-books/interactions/train.parquet',
                'user_id_col': 'userId', 'item_id_col': 'asin',
                'tags_dir': 'data/tags/amazon-books/full',
            },
            'amazon-movies': {
                'train_path': 'data/amazon-movies/interactions/train.parquet',
                'user_id_col': 'userId', 'item_id_col': 'asin',
                'tags_dir': 'data/tags/amazon-movies/full',
            },
        }
        if dataset_name not in DATASET_CONFIGS:
            print(f"  Unknown dataset: {dataset_name}, skipping.")
            continue
        cfg = DATASET_CONFIGS[dataset_name]

        tag_files = sorted(glob.glob(os.path.join(cfg['tags_dir'], '*.parquet')))
        if not tag_files:
            print(f"  No tag files found, skipping.")
            continue

        # Load data and train CF (once)
        print(f"  Loading data and training CF model...")
        train_df = load_df(cfg['train_path'])
        val_df = train_df.head(100)
        test_df = train_df.head(100)
        _, _, _, user2idx, _, item2idx, _ = build_id_mappings(
            train_df, val_df, test_df,
            user_id_col=cfg['user_id_col'], item_id_col=cfg['item_id_col']
        )
        num_items = len(item2idx)

        cf_emb_dict = get_cf_item_emb(train_df, item2idx, cfg['user_id_col'], cfg['item_id_col'])
        emb_dim = next(iter(cf_emb_dict.values())).shape[0]
        cf_embeddings = np.zeros((num_items, emb_dim), dtype=np.float32)
        for item_id, emb in cf_emb_dict.items():
            if item_id in item2idx:
                cf_embeddings[item2idx[item_id]] = emb

        # Pre-compute tag embeddings for all variants (once)
        print(f"  Pre-computing tag embeddings for {len(tag_files)} variants...")
        variant_tag_embeddings = {}
        for tag_file in tag_files:
            variant = os.path.splitext(os.path.basename(tag_file))[0]
            items_df = pd.read_parquet(tag_file)
            tag_embs, tag_item2idx = compute_tag_embeddings(
                items_df, cfg['item_id_col'], 'tags',
                'all-MiniLM-L6-v2', 16, device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            tag_dim = tag_embs.shape[1]
            aligned = np.zeros((num_items, tag_dim), dtype=np.float32)
            for item_id, tag_idx in tag_item2idx.items():
                if item_id in item2idx:
                    aligned[item2idx[item_id]] = tag_embs[tag_idx]
            variant_tag_embeddings[variant] = aligned

        # Load downstream metrics
        downstream_metrics = collect_downstream_metrics(dataset_name, args.results_dir)
        models = sorted(set(m for (_, m) in downstream_metrics.keys()))

        # Sweep panel counts × seeds
        n_total_items = train_df[cfg['item_id_col']].nunique()
        rows = []
        for panel_count in args.panel_counts:
            if panel_count >= n_total_items:
                print(f"  Panel {panel_count}: skipping (>= total items {n_total_items})")
                continue

            panel_size = panel_count / n_total_items
            seed_rhos = {model: [] for model in models}

            for seed in range(args.n_seeds):
                panel_items, item_bins = sample_panel_items(
                    train_df, cfg['item_id_col'],
                    panel_size=panel_size, n_bins=10, seed=42 + seed
                )

                # Compute NO@K for each variant with this panel
                nok_results = {}
                for variant, aligned_tag_embs in variant_tag_embeddings.items():
                    results = evaluate_tag_quality(
                        panel_items, item_bins, cf_embeddings, aligned_tag_embs,
                        item2idx, K=args.K, n_bins=10
                    )
                    nok_results[variant] = results['overall']['mean_recall']

                # Compute correlation per model
                for model in models:
                    model_metrics = {v: m for (v, mod), m in downstream_metrics.items() if mod == model}
                    common = set(nok_results.keys()) & set(model_metrics.keys())
                    if len(common) < 3:
                        continue
                    variants = sorted(common)
                    nok_vals = [nok_results[v] for v in variants]
                    ndcg_vals = [model_metrics[v]['ndcg@10'] for v in variants]
                    rho, _ = stats.spearmanr(nok_vals, ndcg_vals)
                    seed_rhos[model].append(rho)

            # Aggregate across seeds
            for model in models:
                if seed_rhos[model]:
                    rhos = seed_rhos[model]
                    rows.append({
                        'panel_count': panel_count,
                        'model': model,
                        'mean_rho': np.mean(rhos),
                        'std_rho': np.std(rhos),
                        'n_seeds': len(rhos),
                    })

            print(f"  Panel {panel_count} items: done ({args.n_seeds} seeds)")

        if not rows:
            print("  No results to show.")
            continue

        df = pd.DataFrame(rows)
        print(f"\n{'=' * 60}")
        print(f"ABLATION RESULTS: {dataset_name}")
        print(f"{'=' * 60}")
        print(df.to_string(index=False))

        # Save
        csv_path = os.path.join(args.results_dir, f'{dataset_name}_panel_ablation.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved to: {csv_path}")

        # Plot with error bars
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in models:
            model_df = df[df.model == model]
            if len(model_df) > 1:
                ax.errorbar(
                    model_df['panel_count'], model_df['mean_rho'],
                    yerr=model_df['std_rho'],
                    fmt='o-', label=model, markersize=6, capsize=4
                )

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Panel Size (number of items)')
        ax.set_ylabel('Spearman Correlation (NO@K vs NDCG@10)')
        ax.set_title(f'Panel Size Ablation — {dataset_name}\n(mean ± std over {args.n_seeds} random seeds)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(args.results_dir, f'{dataset_name}_panel_ablation.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to: {plot_path}")


if __name__ == '__main__':
    main()
