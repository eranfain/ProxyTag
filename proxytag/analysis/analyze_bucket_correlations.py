#!/usr/bin/env python3
"""
Analyze proxy metric correlations by popularity bucket.

Compares proxy metric predictions with actual hybrid performance across
different item popularity levels (head vs tail).
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def load_proxy_bucket_results(proxy_dir):
    """Load proxy metric results by bucket."""
    # Load main proxy comparison
    proxy_comparison = pd.read_csv(os.path.join(proxy_dir, 'proxy_comparison.csv'))

    # Extract bucket-wise metrics
    bucket_data = []
    for _, row in proxy_comparison.iterrows():
        tag_method = row['tag_method']

        # Get bucket columns (corr_bin_0, recall_bin_0, etc.)
        for col in row.index:
            if col.startswith('corr_bin_'):
                bin_idx = int(col.split('_')[-1])
                recall_col = f'recall_bin_{bin_idx}'

                if pd.notna(row[col]) and recall_col in row.index:
                    bucket_data.append({
                        'tag_method': tag_method,
                        'bucket': bin_idx,
                        'proxy_correlation': row[col],
                        'proxy_recall': row[recall_col] if pd.notna(row[recall_col]) else None
                    })

    return pd.DataFrame(bucket_data)


def load_hybrid_bucket_results(results_dir):
    """Load actual hybrid model bucket results."""
    bucket_files = list(Path(results_dir).glob('*_buckets_eval.csv'))

    all_buckets = []
    for bucket_file in bucket_files:
        # Extract tag method from filename
        # e.g., "amazon-books-hybrid-categories_buckets_eval.csv" -> "categories"
        filename = bucket_file.stem
        tag_method = filename.replace('_buckets_eval', '').split('-hybrid-')[-1]

        # Load bucket results
        df = pd.read_csv(bucket_file)
        df['tag_method'] = tag_method
        all_buckets.append(df)

    if not all_buckets:
        return None

    return pd.concat(all_buckets, ignore_index=True)


def compute_bucket_correlations(proxy_buckets, hybrid_buckets):
    """Compute correlations between proxy and actual performance per bucket."""

    # Merge proxy and hybrid results
    merged = proxy_buckets.merge(
        hybrid_buckets,
        on=['tag_method', 'bucket'],
        how='inner'
    )

    if len(merged) == 0:
        print("   ❌ No matching data between proxy and hybrid results")
        print(f"      Proxy methods: {proxy_buckets['tag_method'].unique()}")
        print(f"      Hybrid methods: {hybrid_buckets['tag_method'].unique()}")
        return None, None

    # Compute correlations per bucket
    bucket_stats = []
    for bucket in sorted(merged['bucket'].unique()):
        bucket_data = merged[merged['bucket'] == bucket]

        if len(bucket_data) < 2:
            continue

        # Correlation between proxy correlation and actual recall
        if len(bucket_data['proxy_correlation'].dropna()) >= 2:
            corr_recall, p_corr_recall = spearmanr(
                bucket_data['proxy_correlation'],
                bucket_data['recall@K']
            )
        else:
            corr_recall, p_corr_recall = np.nan, np.nan

        # Correlation between proxy recall and actual recall
        if len(bucket_data['proxy_recall'].dropna()) >= 2:
            recall_recall, p_recall_recall = spearmanr(
                bucket_data['proxy_recall'],
                bucket_data['recall@K']
            )
        else:
            recall_recall, p_recall_recall = np.nan, np.nan

        bucket_stats.append({
            'bucket': bucket,
            'n_methods': len(bucket_data),
            'spearman_corr_recall': corr_recall,
            'p_corr_recall': p_corr_recall,
            'spearman_recall_recall': recall_recall,
            'p_recall_recall': p_recall_recall,
            'mean_proxy_correlation': bucket_data['proxy_correlation'].mean(),
            'mean_proxy_recall': bucket_data['proxy_recall'].mean(),
            'mean_actual_recall': bucket_data['recall@K'].mean(),
            'mean_actual_ndcg': bucket_data['ndcg@K'].mean(),
        })

    if len(bucket_stats) == 0:
        print("   ❌ Not enough data per bucket for correlation (need >= 2 methods per bucket)")
        return None, None

    return pd.DataFrame(bucket_stats), merged


def plot_bucket_analysis(bucket_stats, merged, output_dir):
    """Create visualization of bucket-wise analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Correlation strength by bucket
    ax = axes[0, 0]
    x = bucket_stats['bucket']
    ax.plot(x, bucket_stats['spearman_corr_recall'], 'o-', label='Proxy Corr ↔ Actual Recall', linewidth=2)
    ax.plot(x, bucket_stats['spearman_recall_recall'], 's-', label='Proxy Recall ↔ Actual Recall', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Popularity Bucket (0=most popular)')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Proxy Metric Predictive Power by Bucket')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean metrics by bucket
    ax = axes[0, 1]
    ax.plot(x, bucket_stats['mean_proxy_correlation'], 'o-', label='Proxy Correlation', linewidth=2)
    ax.plot(x, bucket_stats['mean_proxy_recall'], 's-', label='Proxy Recall', linewidth=2)
    ax.set_xlabel('Popularity Bucket (0=most popular)')
    ax.set_ylabel('Mean Value')
    ax.set_title('Proxy Metrics by Bucket')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Actual performance by bucket
    ax = axes[1, 0]
    ax.plot(x, bucket_stats['mean_actual_recall'], 'o-', label='Recall@K', linewidth=2, color='green')
    ax.plot(x, bucket_stats['mean_actual_ndcg'], 's-', label='NDCG@K', linewidth=2, color='orange')
    ax.set_xlabel('Popularity Bucket (0=most popular)')
    ax.set_ylabel('Mean Value')
    ax.set_title('Actual Hybrid Performance by Bucket')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistical significance
    ax = axes[1, 1]
    significant = bucket_stats['p_corr_recall'] < 0.05
    colors = ['green' if s else 'red' for s in significant]
    ax.bar(x, bucket_stats['spearman_corr_recall'], color=colors, alpha=0.6)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Popularity Bucket (0=most popular)')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Correlation Significance (Green=p<0.05)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'bucket_correlation_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze proxy metric by popularity bucket")

    parser.add_argument('--proxy_dir', type=str, required=True,
                        help='Directory with proxy results (e.g., proxy_results/amazon-books)')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory with hybrid results (e.g., results/amazon-books)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as proxy_dir)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.proxy_dir

    print("="*80)
    print("BUCKET-WISE CORRELATION ANALYSIS")
    print("="*80)

    # Load data
    print("\n1. Loading proxy metric bucket results...")
    proxy_buckets = load_proxy_bucket_results(args.proxy_dir)
    print(f"   Loaded {len(proxy_buckets)} proxy bucket measurements")

    print("\n2. Loading hybrid model bucket results...")
    hybrid_buckets = load_hybrid_bucket_results(args.results_dir)
    if hybrid_buckets is None:
        print("   ❌ No hybrid bucket results found!")
        print("      Make sure hybrid models have been trained.")
        return
    print(f"   Loaded {len(hybrid_buckets)} hybrid bucket measurements")

    # Compute correlations
    print("\n3. Computing bucket-wise correlations...")
    result = compute_bucket_correlations(proxy_buckets, hybrid_buckets)

    if result is None or result[0] is None:
        print("\n⚠️ Bucket analysis skipped - hybrid models still training")
        print("   Run this analysis again after hybrid training completes.")
        return

    bucket_stats, merged = result
    print(f"   Analyzed {len(bucket_stats)} buckets")

    # Print results
    print("\n" + "="*80)
    print("BUCKET-WISE CORRELATION RESULTS")
    print("="*80)
    print("\nBucket 0 = Most Popular Items, Higher Bucket = Less Popular (Tail)")
    print()
    print(bucket_stats[[
        'bucket', 'n_methods',
        'spearman_corr_recall', 'p_corr_recall',
        'mean_actual_recall'
    ]].to_string(index=False))

    # Identify trends
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Where is proxy most predictive?
    best_bucket = bucket_stats.loc[bucket_stats['spearman_corr_recall'].idxmax()]
    worst_bucket = bucket_stats.loc[bucket_stats['spearman_corr_recall'].idxmin()]

    print(f"\nMost predictive bucket: {int(best_bucket['bucket'])} (ρ = {best_bucket['spearman_corr_recall']:.3f})")
    if best_bucket['p_corr_recall'] < 0.05:
        print(f"  ✓ Statistically significant (p = {best_bucket['p_corr_recall']:.4f})")

    print(f"\nLeast predictive bucket: {int(worst_bucket['bucket'])} (ρ = {worst_bucket['spearman_corr_recall']:.3f})")

    # Check if correlation varies with popularity
    head_mean = bucket_stats.iloc[:3]['spearman_corr_recall'].mean()
    tail_mean = bucket_stats.iloc[-3:]['spearman_corr_recall'].mean()

    print(f"\nHead items (buckets 0-2): avg ρ = {head_mean:.3f}")
    print(f"Tail items (last 3 buckets): avg ρ = {tail_mean:.3f}")

    if head_mean > tail_mean + 0.1:
        print("  → Proxy works better for POPULAR items")
    elif tail_mean > head_mean + 0.1:
        print("  → Proxy works better for UNPOPULAR items")
    else:
        print("  → Proxy performance is CONSISTENT across popularity")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    stats_path = os.path.join(args.output_dir, 'bucket_correlation_stats.csv')
    bucket_stats.to_csv(stats_path, index=False)
    print(f"\n✓ Bucket statistics saved: {stats_path}")

    merged_path = os.path.join(args.output_dir, 'bucket_merged_data.csv')
    merged.to_csv(merged_path, index=False)
    print(f"✓ Merged bucket data saved: {merged_path}")

    # Create visualization
    print("\n4. Creating visualizations...")
    plot_bucket_analysis(bucket_stats, merged, args.output_dir)

    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
