#!/usr/bin/env python3
"""
Analyze proxy metric (NO@K) by popularity bucket.

Compares NO@K proxy predictions with actual hybrid performance across
different item popularity levels (head vs tail).
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_proxy_bucket_results(proxy_dir):
    """Load proxy metric results by bucket."""
    # Load main proxy comparison
    proxy_comparison = pd.read_csv(os.path.join(proxy_dir, 'proxy_comparison.csv'))

    # Extract bucket-wise metrics
    bucket_data = []
    for _, row in proxy_comparison.iterrows():
        tag_method = row['tag_method']

        # Get bucket columns (recall_bin_0, etc.)
        for col in row.index:
            if col.startswith('recall_bin_'):
                bin_idx = int(col.split('_')[-1])

                if pd.notna(row[col]):
                    bucket_data.append({
                        'tag_method': tag_method,
                        'bucket': bin_idx,
                        'proxy_recall': row[col]
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


def compute_bucket_analysis(proxy_buckets, hybrid_buckets):
    """Compute analysis between proxy NO@K and actual performance per bucket."""

    # Merge proxy and hybrid results
    merged = proxy_buckets.merge(
        hybrid_buckets,
        on=['tag_method', 'bucket'],
        how='inner'
    )

    if len(merged) == 0:
        print("   No matching data between proxy and hybrid results")
        print(f"      Proxy methods: {proxy_buckets['tag_method'].unique()}")
        print(f"      Hybrid methods: {hybrid_buckets['tag_method'].unique()}")
        return None, None

    # Compute stats per bucket
    bucket_stats = []
    for bucket in sorted(merged['bucket'].unique()):
        bucket_data = merged[merged['bucket'] == bucket]

        if len(bucket_data) < 2:
            continue

        bucket_stats.append({
            'bucket': bucket,
            'n_methods': len(bucket_data),
            'mean_proxy_recall': bucket_data['proxy_recall'].mean(),
            'mean_actual_recall': bucket_data['recall@K'].mean(),
            'mean_actual_ndcg': bucket_data['ndcg@K'].mean(),
        })

    if len(bucket_stats) == 0:
        print("   Not enough data per bucket (need >= 2 methods per bucket)")
        return None, None

    return pd.DataFrame(bucket_stats), merged


def plot_bucket_analysis(bucket_stats, merged, output_dir):
    """Create visualization of bucket-wise analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = bucket_stats['bucket']

    # Plot 1: Proxy NO@K by bucket
    ax = axes[0]
    ax.plot(x, bucket_stats['mean_proxy_recall'], 'o-', label='Proxy NO@K', linewidth=2)
    ax.set_xlabel('Popularity Bucket (0=most popular)')
    ax.set_ylabel('Mean NO@K')
    ax.set_title('Proxy NO@K by Bucket')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Actual performance by bucket
    ax = axes[1]
    ax.plot(x, bucket_stats['mean_actual_recall'], 'o-', label='Recall@K', linewidth=2, color='green')
    ax.plot(x, bucket_stats['mean_actual_ndcg'], 's-', label='NDCG@K', linewidth=2, color='orange')
    ax.set_xlabel('Popularity Bucket (0=most popular)')
    ax.set_ylabel('Mean Value')
    ax.set_title('Actual Hybrid Performance by Bucket')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'bucket_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze proxy metric (NO@K) by popularity bucket")

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
    print("BUCKET-WISE NO@K ANALYSIS")
    print("="*80)

    # Load data
    print("\n1. Loading proxy metric bucket results...")
    proxy_buckets = load_proxy_bucket_results(args.proxy_dir)
    print(f"   Loaded {len(proxy_buckets)} proxy bucket measurements")

    print("\n2. Loading hybrid model bucket results...")
    hybrid_buckets = load_hybrid_bucket_results(args.results_dir)
    if hybrid_buckets is None:
        print("   No hybrid bucket results found!")
        print("      Make sure hybrid models have been trained.")
        return
    print(f"   Loaded {len(hybrid_buckets)} hybrid bucket measurements")

    # Compute analysis
    print("\n3. Computing bucket-wise analysis...")
    result = compute_bucket_analysis(proxy_buckets, hybrid_buckets)

    if result is None or result[0] is None:
        print("\n  Bucket analysis skipped - hybrid models still training")
        print("   Run this analysis again after hybrid training completes.")
        return

    bucket_stats, merged = result
    print(f"   Analyzed {len(bucket_stats)} buckets")

    # Print results
    print("\n" + "="*80)
    print("BUCKET-WISE NO@K RESULTS")
    print("="*80)
    print("\nBucket 0 = Most Popular Items, Higher Bucket = Less Popular (Tail)")
    print()
    print(bucket_stats[[
        'bucket', 'n_methods',
        'mean_proxy_recall', 'mean_actual_recall'
    ]].to_string(index=False))

    # Identify trends
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Where is proxy NO@K highest/lowest?
    best_bucket = bucket_stats.loc[bucket_stats['mean_proxy_recall'].idxmax()]
    worst_bucket = bucket_stats.loc[bucket_stats['mean_proxy_recall'].idxmin()]

    print(f"\nHighest NO@K bucket: {int(best_bucket['bucket'])} (NO@K = {best_bucket['mean_proxy_recall']:.3f})")
    print(f"Lowest NO@K bucket: {int(worst_bucket['bucket'])} (NO@K = {worst_bucket['mean_proxy_recall']:.3f})")

    # Check if NO@K varies with popularity
    head_mean = bucket_stats.iloc[:3]['mean_proxy_recall'].mean()
    tail_mean = bucket_stats.iloc[-3:]['mean_proxy_recall'].mean()

    print(f"\nHead items (buckets 0-2): avg NO@K = {head_mean:.3f}")
    print(f"Tail items (last 3 buckets): avg NO@K = {tail_mean:.3f}")

    if head_mean > tail_mean + 0.05:
        print("  -> Higher overlap for POPULAR items")
    elif tail_mean > head_mean + 0.05:
        print("  -> Higher overlap for UNPOPULAR items")
    else:
        print("  -> NO@K is CONSISTENT across popularity")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    stats_path = os.path.join(args.output_dir, 'bucket_nok_stats.csv')
    bucket_stats.to_csv(stats_path, index=False)
    print(f"\n  Bucket statistics saved: {stats_path}")

    merged_path = os.path.join(args.output_dir, 'bucket_merged_data.csv')
    merged.to_csv(merged_path, index=False)
    print(f"  Merged bucket data saved: {merged_path}")

    # Create visualization
    print("\n4. Creating visualizations...")
    plot_bucket_analysis(bucket_stats, merged, args.output_dir)

    print("\n" + "="*80)
    print("  Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
