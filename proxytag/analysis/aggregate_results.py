#!/usr/bin/env python3
"""
Aggregate all experiment results into a single CSV file.
"""

import json
import os
import pandas as pd
from pathlib import Path

def extract_tag_name(items_data_path):
    """Extract clean tag name from path."""
    filename = Path(items_data_path).name
    # Remove 'tags_' prefix and '.parquet' suffix
    name = filename.replace('tags_', '').replace('.parquet', '')
    return name

def aggregate_results(results_dir='results', compute_stats=False):
    """Aggregate all JSON results into a DataFrame.

    Args:
        results_dir: Directory containing result JSON files
        compute_stats: If True, compute mean/std across runs with same config
    """
    results = []

    # Find all result JSON files
    json_files = list(Path(results_dir).glob('*_results.json'))

    if not json_files:
        print(f"No result files found in {results_dir}/")
        return None

    print(f"Found {len(json_files)} result files\n")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract tag name
        tag_name = extract_tag_name(data['items_data_path'])

        # Create result row
        result = {
            'tag_file': tag_name,
            'job_name': data['job_name'],
            'hidden_dim': data['hidden_dim'],
            'n_heads': data['n_heads'],
            'batch_size': data['batch_size'],
            'lr': data['lr'],
            'max_epochs': data['max_epochs'],
            'embedding_reg': data['embedding_reg'],
            'use_item_id': data['use_item_id'],
            'use_tags': data['use_tags'],
            'train_negatives': data['train_negatives'],
            'test_negatives': data['test_negatives'],
            'K': data['K'],
            f'precision@{data["K"]}': data[f'precision@{data["K"]}'],
            f'recall@{data["K"]}': data[f'recall@{data["K"]}'],
            f'ndcg@{data["K"]}': data[f'ndcg@{data["K"]}'],
            'n_eval': data['n_eval'],
        }

        # Add std if available
        if f'precision@{data["K"]}_std' in data:
            result[f'precision@{data["K"]}_std'] = data[f'precision@{data["K"]}_std']
            result[f'recall@{data["K"]}_std'] = data[f'recall@{data["K"]}_std']
            result[f'ndcg@{data["K"]}_std'] = data[f'ndcg@{data["K"]}_std']

        results.append(result)
        print(f"Loaded: {tag_name} - Recall@{data['K']}: {data[f'recall@{data['K']}']*100:.2f}%")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by recall descending
    df = df.sort_values(f'recall@{df["K"].iloc[0]}', ascending=False)

    return df


def compute_statistics(df):
    """Compute mean and std across runs with same tag_file configuration.

    Groups by tag_file and computes statistics for metric columns.
    """
    if df is None or len(df) == 0:
        return None

    K = df['K'].iloc[0]
    metric_cols = [f'precision@{K}', f'recall@{K}', f'ndcg@{K}']

    # Group by tag_file
    stats = df.groupby('tag_file')[metric_cols].agg(['mean', 'std', 'count'])

    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]

    # Reset index to make tag_file a column
    stats = stats.reset_index()

    # Sort by mean recall descending
    stats = stats.sort_values(f'recall@{K}_mean', ascending=False)

    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result JSON files')
    parser.add_argument('--compute_stats', action='store_true',
                       help='Compute mean/std across runs with same config')
    args = parser.parse_args()

    print("="*80)
    print("AGGREGATING EXPERIMENT RESULTS")
    print("="*80)
    print()

    df = aggregate_results(args.results_dir)

    if df is None:
        return

    # Save all results to CSV
    output_file = f'{args.results_dir}/all_results.csv'
    df.to_csv(output_file, index=False)

    print()
    print("="*80)
    print(f"All results saved to: {output_file}")
    print("="*80)
    print()

    K = df['K'].iloc[0]

    # Check if std columns exist
    has_std = f'recall@{K}_std' in df.columns

    # Print summary table
    print("SUMMARY (sorted by recall):")
    print("-" * 80)

    if has_std:
        summary_cols = ['tag_file', f'recall@{K}', f'recall@{K}_std', f'ndcg@{K}', f'ndcg@{K}_std', f'precision@{K}', f'precision@{K}_std']
        summary = df[summary_cols].copy()
        summary[f'recall@{K}'] = summary[f'recall@{K}'] * 100
        summary[f'recall@{K}_std'] = summary[f'recall@{K}_std'] * 100
        summary[f'ndcg@{K}'] = summary[f'ndcg@{K}'] * 100
        summary[f'ndcg@{K}_std'] = summary[f'ndcg@{K}_std'] * 100
        summary[f'precision@{K}'] = summary[f'precision@{K}'] * 100
        summary[f'precision@{K}_std'] = summary[f'precision@{K}_std'] * 100
        summary.columns = ['Tag File', 'Recall@10 (%)', 'Recall Std (%)', 'NDCG@10 (%)', 'NDCG Std (%)', 'Precision@10 (%)', 'Precision Std (%)']
    else:
        summary_cols = ['tag_file', f'recall@{K}', f'ndcg@{K}', f'precision@{K}']
        summary = df[summary_cols].copy()
        summary[f'recall@{K}'] = summary[f'recall@{K}'] * 100
        summary[f'ndcg@{K}'] = summary[f'ndcg@{K}'] * 100
        summary[f'precision@{K}'] = summary[f'precision@{K}'] * 100
        summary.columns = ['Tag File', 'Recall@10 (%)', 'NDCG@10 (%)', 'Precision@10 (%)']

    print(summary.to_string(index=False))
    print()

    # Compute statistics if requested
    if args.compute_stats:
        print()
        print("="*80)
        print("STATISTICS ACROSS RUNS (Mean ± Std)")
        print("="*80)
        print()

        stats_df = compute_statistics(df)

        if stats_df is not None:
            # Save statistics to CSV
            stats_file = f'{args.results_dir}/statistics.csv'
            stats_df.to_csv(stats_file, index=False)
            print(f"Statistics saved to: {stats_file}\n")

            # Print statistics table (convert to percentages)
            display_stats = stats_df.copy()
            for col in stats_df.columns:
                if '_mean' in col or '_std' in col:
                    display_stats[col] = display_stats[col] * 100

            # Format output nicely
            print("Statistics by Tag File:")
            print("-" * 80)
            for _, row in display_stats.iterrows():
                tag = row['tag_file']
                n = int(row[f'recall@{K}_count'])
                recall_mean = row[f'recall@{K}_mean']
                recall_std = row[f'recall@{K}_std']
                ndcg_mean = row[f'ndcg@{K}_mean']
                ndcg_std = row[f'ndcg@{K}_std']
                prec_mean = row[f'precision@{K}_mean']
                prec_std = row[f'precision@{K}_std']

                print(f"\n{tag} (n={n} runs):")
                print(f"  Recall@{K}:    {recall_mean:6.2f}% ± {recall_std:5.2f}%")
                print(f"  NDCG@{K}:      {ndcg_mean:6.2f}% ± {ndcg_std:5.2f}%")
                print(f"  Precision@{K}: {prec_mean:6.2f}% ± {prec_std:5.2f}%")

            print()
        else:
            print("No statistics to compute (insufficient data)")
            print()

if __name__ == "__main__":
    main()
