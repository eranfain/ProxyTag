#!/usr/bin/env python3
"""
Temporal train/val/test splitting for recommendation datasets.

Splits interaction data into 80/10/10 train/val/test based on timestamps.
Supports both MovieLens and Amazon Books formats.
"""

import argparse
import os
import pandas as pd
from datetime import datetime


def split_temporal(ratings_df, train_ratio=0.8, val_ratio=0.1):
    """
    Splits interaction data temporally based on timestamps.

    Args:
        ratings_df: DataFrame with user/item interactions and timestamp column
        train_ratio: Proportion for training (default: 0.8)
        val_ratio: Proportion for validation (default: 0.1)

    Returns:
        train_df, val_df, test_df
    """
    print(f"Original Dataset Size: {len(ratings_df):,}")

    # Sort by timestamp
    print("Sorting data by timestamp...")
    ratings_df = ratings_df.sort_values(by='timestamp').reset_index(drop=True)

    # Find cutoff indices
    n_samples = len(ratings_df)
    idx_train = int(n_samples * train_ratio)
    idx_val = int(n_samples * (train_ratio + val_ratio))

    # Extract timestamps at cutoffs
    timestamp_train = ratings_df.iloc[idx_train]['timestamp']
    timestamp_val = ratings_df.iloc[idx_val]['timestamp']

    # Convert to readable dates
    date_train = datetime.fromtimestamp(timestamp_train)
    date_val = datetime.fromtimestamp(timestamp_val)

    print(f"\n--- Split Boundaries ---")
    print(f"Train Cutoff: {date_train} (Timestamp: {timestamp_train})")
    print(f"Val Cutoff: {date_val} (Timestamp: {timestamp_val})")

    # Create splits
    train = ratings_df[ratings_df['timestamp'] < timestamp_train].copy()
    val = ratings_df[
        (ratings_df['timestamp'] >= timestamp_train) &
        (ratings_df['timestamp'] < timestamp_val)
    ].copy()
    test = ratings_df[ratings_df['timestamp'] >= timestamp_val].copy()

    # Verify split ratios
    print(f"\n--- Final Split Sizes ---")
    print(f"Train: {len(train):,} ({len(train)/n_samples:.2%})")
    print(f"Val:   {len(val):,} ({len(val)/n_samples:.2%})")
    print(f"Test:  {len(test):,} ({len(test)/n_samples:.2%})")

    return train, val, test


def prepare_tag_files(data_dir, output_dir, tag_column_name='tags'):
    """
    Convert full_data_*.parquet files to tags_*.parquet format.

    Extracts only item ID and tags columns, renames tag column to 'tags'.
    Saves to tags/full/ subdirectory.
    """
    print(f"\n--- Processing Tag Files ---")

    # Create tags directory structure
    tags_full_dir = os.path.join(output_dir, 'tags', 'full')
    os.makedirs(tags_full_dir, exist_ok=True)

    processed = 0

    for filename in os.listdir(data_dir):
        if filename.startswith("full_data") and filename.endswith(".parquet"):
            # Extract tag type from filename
            tag_type = filename.replace("full_data_", "").replace(".parquet", "")

            # Load and process
            df_tags = pd.read_parquet(os.path.join(data_dir, filename))

            # Determine item ID column (movieId or asin)
            if 'movieId' in df_tags.columns:
                item_col = 'movieId'
            elif 'asin' in df_tags.columns:
                item_col = 'asin'
            else:
                print(f"  Warning: No item ID column found in {filename}, skipping")
                continue

            # Rename tag column to 'tags'
            if tag_type in df_tags.columns:
                df_tags = df_tags.rename(columns={tag_type: 'tags'})

            # Save with only item ID and tags to tags/full/
            output_file = os.path.join(tags_full_dir, f"tags_{tag_type}.parquet")
            df_tags[[item_col, 'tags']].to_parquet(output_file, index=False)
            print(f"  Created: {output_file}")
            processed += 1

    print(f"Processed {processed} tag files")
    print(f"✓ Saved to {tags_full_dir}/")
    print(f"\nNote: Panel tag files will be created during proxy metric evaluation")


def main():
    parser = argparse.ArgumentParser(description="Prepare recommendation data")
    parser.add_argument('--interactions_path', type=str, required=True,
                        help='Path to interactions file (e.g., ratings.csv)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for splits')
    parser.add_argument('--user_id_col', type=str, default='userId',
                        help='User ID column name')
    parser.add_argument('--item_id_col', type=str, default='movieId',
                        help='Item ID column name')
    parser.add_argument('--filter_items', type=str, default=None,
                        help='Optional: filter to items in this parquet file')
    parser.add_argument('--process_tags', action='store_true',
                        help='Process full_data_*.parquet files to tags format')
    parser.add_argument('--tags_dir', type=str, default=None,
                        help='Directory with full_data_*.parquet files')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set proportion (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set proportion (default: 0.1)')
    parser.add_argument('--create_tiny', action='store_true',
                        help='Create tiny subset with val/test users only')

    args = parser.parse_args()

    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    interactions_dir = os.path.join(args.output_dir, 'interactions')
    os.makedirs(interactions_dir, exist_ok=True)

    # Load interactions
    print(f"Loading interactions from: {args.interactions_path}")
    if args.interactions_path.endswith('.parquet'):
        ratings = pd.read_parquet(args.interactions_path)
    else:
        ratings = pd.read_csv(args.interactions_path)

    # Filter to specific items if requested
    if args.filter_items:
        print(f"Filtering to items in: {args.filter_items}")
        items = pd.read_parquet(args.filter_items)
        ratings = ratings[ratings[args.item_id_col].isin(items[args.item_id_col])].reset_index(drop=True)

    print(f"Total interactions: {len(ratings):,}")
    print(f"Avg interactions per user: {len(ratings)/ratings[args.user_id_col].nunique():.2f}")

    # Temporal split
    train_df, val_df, test_df = split_temporal(ratings, args.train_ratio, args.val_ratio)

    # Save main splits (only user and item columns)
    cols = [args.user_id_col, args.item_id_col]
    train_df[cols].to_parquet(os.path.join(interactions_dir, 'train.parquet'), index=False)
    val_df[cols].to_parquet(os.path.join(interactions_dir, 'val.parquet'), index=False)
    test_df[cols].to_parquet(os.path.join(interactions_dir, 'test.parquet'), index=False)

    print(f"\n✓ Saved splits to {interactions_dir}/")

    # Create tiny subset if requested
    if args.create_tiny:
        print("\n--- Creating Tiny Subset ---")
        tiny_users = set(val_df[args.user_id_col].unique())
        tiny_users.update(test_df[args.user_id_col].unique())

        train_tiny = train_df[train_df[args.user_id_col].isin(tiny_users)][cols]
        val_tiny = val_df[val_df[args.user_id_col].isin(tiny_users)][cols]
        test_tiny = test_df[test_df[args.user_id_col].isin(tiny_users)][cols]

        train_tiny.to_parquet(os.path.join(interactions_dir, 'train_tiny.parquet'), index=False)
        val_tiny.to_parquet(os.path.join(interactions_dir, 'val_tiny.parquet'), index=False)
        test_tiny.to_parquet(os.path.join(interactions_dir, 'test_tiny.parquet'), index=False)

        print(f"  Train tiny: {len(train_tiny):,}")
        print(f"  Val tiny: {len(val_tiny):,}")
        print(f"  Test tiny: {len(test_tiny):,}")
        print(f"✓ Saved tiny splits to {interactions_dir}/")

    # Process tag files if requested
    if args.process_tags:
        tags_dir = args.tags_dir or args.output_dir
        prepare_tag_files(tags_dir, args.output_dir)

    print("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()
