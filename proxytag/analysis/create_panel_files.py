#!/usr/bin/env python3
"""
Create panel tag files from full tag files.

Samples a panel of items (typically head items) and creates corresponding
panel tag files for fast proxy metric evaluation.
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from proxytag.core.data import load_df


def sample_panel_items(train_df, item_col, sample_mode='head', panel_size=0.05,
                       head_percentile=0.3, n_bins=10, seed=42):
    """
    Sample panel items based on specified mode.

    Args:
        train_df: Training interactions
        item_col: Item ID column name
        sample_mode: 'head', 'stratified', or 'all'
        panel_size: Fraction of items to sample
        head_percentile: Top X% for head sampling
        n_bins: Number of bins for stratified sampling
        seed: Random seed

    Returns:
        panel_items: Array of sampled item IDs
        item_bins: Dict mapping item ID to popularity bin
    """
    print(f"--- Sampling Panel Items ({sample_mode} mode) ---")

    # Compute item frequencies
    item_counts = train_df[item_col].value_counts()
    n_items = len(item_counts)

    np.random.seed(seed)

    if sample_mode == 'head':
        # Sample from top X% most popular items
        n_head = max(1, int(n_items * head_percentile))
        head_items = item_counts.head(n_head).index.values

        # Sample panel_size fraction of ALL items, but from head pool only
        n_sample = max(100, int(n_items * panel_size))
        n_sample = min(n_sample, len(head_items))  # Cap at available head items
        panel_items = np.random.choice(head_items, size=n_sample, replace=False)

        # All items in same bin for head mode
        item_bins = {item: 0 for item in panel_items}

        print(f"  Total items: {n_items:,}")
        print(f"  Head items (top {head_percentile:.0%}): {n_head:,}")
        print(f"  Panel sample: {len(panel_items):,} ({len(panel_items)/n_items:.2%})")

    elif sample_mode == 'stratified':
        # Assign items to popularity bins
        item_bins = {}
        for i, (item_id, count) in enumerate(item_counts.items()):
            bin_idx = min(i * n_bins // n_items, n_bins - 1)
            item_bins[item_id] = bin_idx

        # Stratified sampling
        panel_items = []
        for bin_idx in range(n_bins):
            bin_items = [item for item, b in item_bins.items() if b == bin_idx]
            n_sample = max(1, int(len(bin_items) * panel_size))
            sampled = np.random.choice(bin_items, size=n_sample, replace=False)
            panel_items.extend(sampled)

        panel_items = np.array(panel_items)

        print(f"  Total items: {n_items:,}")
        print(f"  Panel items: {len(panel_items):,} ({len(panel_items)/n_items:.2%})")
        print(f"  Bins: {n_bins}")

    else:  # 'all'
        n_sample = max(100, int(n_items * panel_size))
        panel_items = np.random.choice(item_counts.index.values, size=n_sample, replace=False)
        item_bins = {item: 0 for item in panel_items}

        print(f"  Total items: {n_items:,}")
        print(f"  Panel items: {len(panel_items):,} ({len(panel_items)/n_items:.2%})")

    return panel_items, item_bins


def create_panel_files(full_tag_dir, panel_tag_dir, panel_items, item_col):
    """
    Create panel tag files from full tag files.

    Args:
        full_tag_dir: Directory with full tag files
        panel_tag_dir: Output directory for panel files
        panel_items: Array of panel item IDs
        item_col: Item ID column name
    """
    print(f"\n--- Creating Panel Tag Files ---")

    os.makedirs(panel_tag_dir, exist_ok=True)

    panel_items_set = set(panel_items)
    processed = 0
    skipped = 0

    for tag_file in Path(full_tag_dir).glob('tags_*.parquet'):
        # Load full file
        try:
            df = pd.read_parquet(tag_file)
        except FileNotFoundError:
            print(f"  ❌ Skipped: {tag_file} (not found)")
            skipped += 1
            continue
        except Exception as e:
            print(f"  ❌ Skipped: {tag_file} (corrupted or invalid: {type(e).__name__})")
            skipped += 1
            continue

        # Filter to panel items
        panel_df = df[df[item_col].isin(panel_items_set)].copy()

        # Save panel file
        output_file = os.path.join(panel_tag_dir, tag_file.name)
        panel_df.to_parquet(output_file, index=False)

        print(f"  Created: {output_file}")
        print(f"    Items: {len(panel_df):,} / {len(df):,} ({len(panel_df)/len(df):.2%})")
        processed += 1

    print(f"\n✓ Created {processed} panel tag files")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} corrupted/invalid files")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Create panel tag files for proxy evaluation")

    # Paths
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset directory (e.g., data/amazon-books)')
    parser.add_argument('--train_path', type=str, default=None,
                        help='Training interactions (default: {dataset_dir}/interactions/train.parquet)')
    parser.add_argument('--interactions_suffix', type=str, default='',
                        help='Suffix for interactions files (e.g., "_tiny" for train_tiny.parquet)')

    # Column names
    parser.add_argument('--item_id_col', type=str, default='movieId')

    # Sampling parameters
    parser.add_argument('--sample_mode', type=str, default='head',
                        choices=['head', 'stratified', 'all'],
                        help='Sampling mode: head (top items), stratified, or all')
    parser.add_argument('--panel_size', type=float, default=0.05,
                        help='Fraction of items to sample (default: 0.05)')
    parser.add_argument('--head_percentile', type=float, default=0.3,
                        help='Top X percentile for head mode (default: 0.3)')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of bins for stratified mode (default: 10)')

    # Other
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print("="*80)
    print("CREATE PANEL TAG FILES")
    print("="*80)
    print()

    # Paths
    dataset_dir = args.dataset_dir
    suffix = args.interactions_suffix
    train_path = args.train_path or os.path.join(dataset_dir, 'interactions', f'train{suffix}.parquet')
    full_tag_dir = os.path.join(dataset_dir, 'tags', 'full')  # Always 'full', tags are for full catalog
    panel_tag_dir = os.path.join(dataset_dir, 'tags', f'panel{suffix}')  # Panel gets suffix

    # Validate paths
    if not os.path.exists(train_path):
        print(f"❌ Error: Training file not found: {train_path}")
        return

    # Full tag directory is optional - we just create panel items list if it doesn't exist
    if not os.path.exists(full_tag_dir):
        print(f"⚠️  Warning: Full tag directory not found: {full_tag_dir}")
        print(f"   Will only create panel items list (not panel tag files)")
        print()

    # Load training data
    print(f"Loading training data from: {train_path}")
    train_df = load_df(train_path)
    print(f"  Interactions: {len(train_df):,}")
    print(f"  Items: {train_df[args.item_id_col].nunique():,}")
    print()

    # Sample panel items
    panel_items, item_bins = sample_panel_items(
        train_df,
        args.item_id_col,
        sample_mode=args.sample_mode,
        panel_size=args.panel_size,
        head_percentile=args.head_percentile,
        n_bins=args.n_bins,
        seed=args.seed
    )

    # Save panel items list
    panel_items_file = os.path.join(dataset_dir, f'panel_items{suffix}.parquet')
    panel_df = pd.DataFrame({args.item_id_col: panel_items})
    panel_df.to_parquet(panel_items_file, index=False)
    print(f"\n✓ Saved panel items to: {panel_items_file}")

    # Create panel tag files (if full tag dir exists)
    if os.path.exists(full_tag_dir):
        n_created = create_panel_files(
            full_tag_dir,
            panel_tag_dir,
            panel_items,
            args.item_id_col
        )
    else:
        print(f"\n⚠️ Full tag directory not found: {full_tag_dir}")
        print(f"   Skipping panel tag file creation.")
        print(f"   Generate tags for panel first using the panel_items.parquet file.")
        n_created = 0

    if n_created > 0:
        print()
        print("="*80)
        print("✓ PANEL CREATION COMPLETE")
        print("="*80)
        print()
        print(f"Panel items file: {panel_items_file}")
        print(f"Panel items: {len(panel_items):,}")
        print(f"Tag files created: {n_created}")
        print()
        print("Next step: Generate tags for panel items")
        print(f"  1. Filter metadata to panel:")
        print(f"     python3 -c \"import pandas as pd; \\")
        print(f"       meta = pd.read_parquet('metadata.parquet'); \\")
        print(f"       panel = pd.read_parquet('{panel_items_file}'); \\")
        print(f"       panel_meta = meta[meta['{args.item_id_col}'].isin(panel['{args.item_id_col}'])]; \\")
        print(f"       panel_meta.to_parquet('panel_metadata.parquet')\"")
        print()
        print(f"  2. Generate tags for panel:")
        print(f"     python generate_tags.py --input_path panel_metadata.parquet ...")
    else:
        print()
        print("="*80)
        print("✓ PANEL ITEMS SAMPLED")
        print("="*80)
        print()
        print(f"Panel items file: {panel_items_file}")
        print(f"Panel items: {len(panel_items):,}")
        panel_metadata_path = os.path.join(dataset_dir, "panel_metadata.parquet")

        print()
        print("Next step: Filter metadata and generate tags")
        print(f"  1. Filter metadata to panel:")
        print(f"     python3 -c \"import pandas as pd; \\")
        print(f"       meta = pd.read_parquet('your_metadata.parquet'); \\")
        print(f"       panel = pd.read_parquet('{panel_items_file}'); \\")
        print(f"       panel_meta = meta[meta['{args.item_id_col}'].isin(panel['{args.item_id_col}'])]; \\")
        print(f"       panel_meta.to_parquet('{panel_metadata_path}')\"")
        print()
        print(f"  2. Generate tags for panel:")
        print(f"     python generate_tags.py \\")
        print(f"       --input_path {panel_metadata_path} \\")
        print(f"       --output_dir {panel_tag_dir}")
        print()
        print(f"  3. Then evaluate with proxy metric:")
        print(f"     python compare_tag_methods.py --panel_tag_dir {panel_tag_dir} ...")


if __name__ == "__main__":
    main()
