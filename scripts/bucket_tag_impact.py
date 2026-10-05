#!/usr/bin/env python3
"""
Per-bucket analysis: NDCG@10 improvement from tags by item popularity.

For each model (AutoInt, DCNv2, UniSRec) x dataset (amazon-books, amazon-movies):
  1. Load no-tags bucket CSV
  2. Find the best tag variant (highest overall NDCG@10) and load its bucket CSV
  3. Compute per-bucket NDCG@10 improvement
  4. Generate a figure showing that cold-start/long-tail items benefit more from tags
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = 'results'
FIGURES_DIR = 'paper/figures'

DATASETS = ['amazon-books', 'amazon-movies']
DATASET_DISPLAY = {'amazon-books': 'Amazon-Books', 'amazon-movies': 'Amazon-Movies'}

MODELS = {
    'autoint': 'AutoInt',
    'dcnv2': 'DCNv2',
}

BUCKET_LABELS = {
    0: 'Cold',
    1: 'Top 10%',
    2: '10-20%',
    3: '20-30%',
    4: '30-40%',
    5: '40-50%',
    6: '50-60%',
    7: '60-70%',
    8: '70-80%',
    9: '80-90%',
    10: '90-100%',
}


def find_best_variant(model_key, dataset):
    """Find the tag variant with highest NDCG@10 for a model/dataset combo."""
    pattern = os.path.join(RESULTS_DIR, f'{model_key}_*_{dataset}_results.json')
    result_files = glob.glob(pattern)

    best_ndcg = -1
    best_variant = None
    for rf in result_files:
        job_name = os.path.basename(rf).replace('_results.json', '')
        if 'no_tags' in job_name:
            continue
        with open(rf) as f:
            data = json.load(f)
        ndcg = data.get('test_metrics', {}).get('ndcg@10', 0)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_variant = job_name
    return best_variant, best_ndcg


def load_bucket_csv(job_name):
    """Load a bucket evaluation CSV."""
    path = os.path.join(RESULTS_DIR, f'{job_name}_buckets_eval.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_improvement(no_tags_df, best_tags_df, metric='ndcg@10'):
    """Compute per-bucket improvement (absolute and relative)."""
    merged = no_tags_df.merge(best_tags_df, on='bucket', suffixes=('_notags', '_tags'))
    merged['abs_improvement'] = merged[f'{metric}_tags'] - merged[f'{metric}_notags']
    merged['rel_improvement'] = np.where(
        merged[f'{metric}_notags'] > 1e-8,
        merged['abs_improvement'] / merged[f'{metric}_notags'] * 100,
        np.nan
    )
    return merged


def generate_figure(all_data):
    """Generate per-bucket figure: NDCG@10 with vs without tags, one subplot per model-dataset."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 7,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    # Collect which models have data
    available_models = []
    for model_key, model_display in MODELS.items():
        if any((model_key, ds) in all_data for ds in DATASETS):
            available_models.append((model_key, model_display))

    n_models = len(available_models)
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(n_models, n_datasets, figsize=(7, 1.8 * n_models + 0.4),
                              squeeze=False)

    c_notags = '#888888'
    c_tags = '#2E86C1'

    for row, (model_key, model_display) in enumerate(available_models):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            key = (model_key, dataset)

            if key not in all_data:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            merged = all_data[key]
            merged = merged[merged['bucket'] > 0]

            x = merged['bucket'].values
            y_notags = merged['ndcg@10_notags'].values
            y_tags = merged['ndcg@10_tags'].values

            ax.plot(x, y_notags, label='No tags', color=c_notags,
                    marker='x', markersize=4, linewidth=1.2, linestyle='--', alpha=0.7)
            ax.plot(x, y_tags, label='Best tags', color=c_tags,
                    marker='o', markersize=4, linewidth=1.3, alpha=0.85)

            ax.fill_between(x, y_notags, y_tags, alpha=0.12, color=c_tags,
                            where=y_tags >= y_notags)
            ax.fill_between(x, y_notags, y_tags, alpha=0.12, color='#E74C3C',
                            where=y_tags < y_notags)

            if row == n_models - 1:
                ax.set_xlabel('Popularity bucket\n(1=popular, 10=long tail)', fontsize=7)
            if col == 0:
                ax.set_ylabel('NDCG@10', fontsize=9)

            if row == 0:
                ax.set_title(DATASET_DISPLAY[dataset], fontsize=11, fontweight='bold')

            ax.text(0.98, 0.95, model_display, transform=ax.transAxes,
                    fontsize=8, fontweight='bold', ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8, edgecolor='#cccccc', linewidth=0.5))

            ax.set_xticks(range(1, 11))
            ax.set_yscale('log')
            ax.set_ylim(bottom=5e-4)
            ax.grid(True, alpha=0.25, linewidth=0.5, which='both')

    # Single shared legend outside the axes
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=8,
               framealpha=0.9, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, 1.03))

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, 'bucket_tag_impact.pdf')
    fig.savefig(fig_path, format='pdf')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


def main():
    all_data = {}

    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        for model_key, model_display in MODELS.items():
            no_tags_job = f'{model_key}_no_tags_{dataset}'
            no_tags_df = load_bucket_csv(no_tags_job)
            if no_tags_df is None:
                print(f"  {model_display}: no-tags bucket CSV not found, skipping")
                continue

            best_variant, best_ndcg = find_best_variant(model_key, dataset)
            if best_variant is None:
                print(f"  {model_display}: no tag variant results found, skipping")
                continue

            best_tags_df = load_bucket_csv(best_variant)
            if best_tags_df is None:
                print(f"  {model_display}: best variant bucket CSV not found ({best_variant}), skipping")
                continue

            merged = compute_improvement(no_tags_df, best_tags_df)

            print(f"\n  {model_display}: best variant = {best_variant} (NDCG@10={best_ndcg:.4f})")
            print(f"  {'Bucket':>8} {'No Tags':>10} {'Best Tags':>10} {'Abs Impr':>10} {'Rel Impr':>10}")
            for _, row in merged.iterrows():
                b = int(row['bucket'])
                label = BUCKET_LABELS.get(b, str(b))
                nt = row['ndcg@10_notags']
                bt = row['ndcg@10_tags']
                ai = row['abs_improvement']
                ri = row['rel_improvement']
                print(f"  {label:>8} {nt:10.4f} {bt:10.4f} {ai:+10.4f} {ri:+9.1f}%")

            all_data[(model_key, dataset)] = merged

    if all_data:
        generate_figure(all_data)

        # Also save raw data
        csv_path = os.path.join(RESULTS_DIR, 'bucket_tag_impact.csv')
        rows = []
        for (model_key, dataset), merged in all_data.items():
            for _, row in merged.iterrows():
                rows.append({
                    'model': MODELS[model_key],
                    'dataset': dataset,
                    'bucket': int(row['bucket']),
                    'ndcg_notags': row['ndcg@10_notags'],
                    'ndcg_best_tags': row['ndcg@10_tags'],
                    'abs_improvement': row['abs_improvement'],
                    'rel_improvement': row['rel_improvement'],
                })
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nRaw data saved: {csv_path}")
    else:
        print("\nNo data available for any model/dataset combo.")


if __name__ == '__main__':
    main()
