#!/usr/bin/env python3
"""
Re-evaluate models with fixed evaluation (random tie-breaking).

This will load existing checkpoints and re-run evaluation with the updated
eval.py that includes seeded random tie-breaking.
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd

# Project imports
from proxytag.core.data import build_id_mappings, build_user_history, NegativeSampler
from proxytag.core.model import HybridRecLightning
from proxytag.core.eval import eval_split_fast


def reevaluate_model(result_file, args):
    """Re-evaluate one model."""
    with open(result_file, 'r') as f:
        result = json.load(f)

    job_name = result['job_name']
    checkpoint_path = f"checkpoints/{job_name}_best.ckpt"

    if not Path(checkpoint_path).exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return None

    print(f"\n{'='*80}")
    print(f"Re-evaluating: {job_name}")
    print(f"{'='*80}")

    # Load the model
    model = HybridRecLightning.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Load data (we need test split and mappings)
    print("Loading data...")

    train_df = pd.read_parquet(args.train_path)
    test_df = pd.read_parquet(args.test_path)

    # Build mappings
    train_mapped, _, test_mapped, user2idx, idx2user, item2idx, idx2item = build_id_mappings(
        train_df=train_df,
        val_df=None,
        test_df=test_df,
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col,
    )

    # Build user history
    all_interactions = pd.concat([train_mapped, test_mapped], ignore_index=True)
    user_history = build_user_history(all_interactions)

    # Create negative sampler for test
    n_items = len(item2idx)
    neg_sampler_test = NegativeSampler(
        user_history=user_history,
        n_items=n_items,
        seed=args.seed + 1,
        max_attempts=args.max_attempts,
    )

    # Run evaluation
    print("Running evaluation with fixed tie-breaking...")

    new_results = eval_split_fast(
        model=model,
        df=test_mapped,
        neg_sampler=neg_sampler_test,
        n_items=n_items,
        n_neg=result['test_negatives'],
        K=result['K'],
        seed=args.seed,
        compute_loss=False,
    )

    print(f"\nOriginal results:")
    print(f"  Precision@{result['K']}: {result['precision@10']:.6f}")
    print(f"  Recall@{result['K']}: {result['recall@10']:.6f}")
    print(f"  NDCG@{result['K']}: {result['ndcg@10']:.6f}")

    print(f"\nNew results (with tie-breaking):")
    print(f"  Precision@{result['K']}: {new_results['precision@K']:.6f}")
    print(f"  Recall@{result['K']}: {new_results['recall@K']:.6f}")
    print(f"  NDCG@{result['K']}: {new_results['ndcg@K']:.6f}")

    # Check if this fixes equal scores bug
    old_recall = result['recall@10']
    new_recall = new_results['recall@K']

    if old_recall > 0.9 and new_recall < 0.1:
        print(f"\n✅ EQUAL SCORES BUG FIXED!")
        print(f"   Recall dropped from {old_recall:.1%} to {new_recall:.1%}")
        print(f"   (Expected ~1% for broken models due to random selection)")
    elif old_recall > 0.9 and new_recall > 0.9:
        print(f"\n⚠️  Still showing high recall - may not be equal scores bug")
    else:
        print(f"\n✓ Model was already working correctly")

    return new_results


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate models with fixed tie-breaking")

    # Data paths
    parser.add_argument('--train_path', type=str, default='data/amazon-books/interactions/train.parquet')
    parser.add_argument('--test_path', type=str, default='data/amazon-books/interactions/test.parquet')
    parser.add_argument('--results_dir', type=str, default='results/amazon-books')

    # Column names
    parser.add_argument('--user_id_col', type=str, default='userId')
    parser.add_argument('--item_id_col', type=str, default='asin')

    # Eval params
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_attempts', type=int, default=100)

    # Which models to re-evaluate
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to re-evaluate (e.g., kar llm_rec_orig)')

    args = parser.parse_args()

    # Find result files
    results_dir = Path(args.results_dir)
    result_files = list(results_dir.glob('*_results.json'))

    if args.models:
        result_files = [f for f in result_files if any(m in f.stem for m in args.models)]

    if not result_files:
        print(f"❌ No result files found")
        return

    print(f"Found {len(result_files)} models to re-evaluate")

    # Re-evaluate each
    for result_file in sorted(result_files):
        try:
            reevaluate_model(result_file, args)
        except Exception as e:
            print(f"\n❌ Error re-evaluating {result_file.stem}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
