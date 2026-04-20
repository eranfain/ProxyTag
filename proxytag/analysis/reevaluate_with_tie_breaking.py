#!/usr/bin/env python3
"""
Re-evaluate all models with fixed tie-breaking to diagnose equal scores bug.

Run this on EC2 where checkpoints exist.

Expected results:
- If model truly has equal scores: recall will drop to ~0.1% (random chance: 1/1001)
- If model has good scores: recall will stay similar (tie-breaking doesn't affect)

Usage:
    python reevaluate_with_tie_breaking.py \
        --train_path data/amazon-books/interactions/train.parquet \
        --test_path data/amazon-books/interactions/test.parquet \
        --results_dir results/amazon-books
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import torch

# Project imports
from proxytag.core.data import build_id_mappings, build_user_history, NegativeSampler
from proxytag.core.model import HybridRecLightning
from proxytag.core.eval import eval_split_fast
from proxytag.core.tags import build_item_tag_tensors


def reevaluate_model(result_file, train_df, test_df, args):
    """Re-evaluate one model with tie-breaking."""
    with open(result_file, 'r') as f:
        result = json.load(f)

    job_name = result['job_name']
    # Checkpoint path structure: checkpoints/{dataset}/{job_name}/best.ckpt
    # Extract dataset from results_dir (e.g., results/amazon-books -> amazon-books)
    dataset = Path(result_file).parent.name
    checkpoint_path = f"checkpoints/{dataset}/{job_name}/best.ckpt"

    if not Path(checkpoint_path).exists():
        print(f"⚠️  Skipping {job_name} - checkpoint not found: {checkpoint_path}")
        return None

    print(f"\n{'='*80}")
    print(f"Re-evaluating: {job_name}")
    print(f"{'='*80}")

    try:
        # Load model
        print("Loading checkpoint...")
        model = HybridRecLightning.load_from_checkpoint(checkpoint_path)
        model.eval()

        # Build mappings
        print("Building ID mappings...")
        train_mapped, _, test_mapped, user2idx, idx2user, item2idx, idx2item = build_id_mappings(
            train_df=train_df,
            val_df=None,
            test_df=test_df,
            user_id_col=args.user_id_col,
            item_id_col=args.item_id_col,
        )

        # Load tag embeddings if model uses tags
        if result.get('use_tags', True) and result.get('items_data_path'):
            print("Loading tag embeddings...")
            items_data_path = result['items_data_path']
            if not Path(items_data_path).exists():
                print(f"⚠️  Warning: Items data not found: {items_data_path}")
                print("   Continuing without tags (will fail if model requires them)")
            else:
                device = next(model.parameters()).device

                # Get hyperparameters from result JSON with defaults
                st_model = result.get('sentence_transformers_model', 'all-MiniLM-L6-v2')
                max_tags = result.get('max_tags', 36)

                item_tag_embs, item_tag_mask = build_item_tag_tensors(
                    items_df_path=items_data_path,
                    item2idx=item2idx,
                    model_name=st_model,
                    max_tags=max_tags,
                    device=str(device),
                    item_id_col=args.item_id_col,
                )
                model.item_tag_embs = item_tag_embs
                model.item_tag_mask = item_tag_mask
                print(f"   Loaded tags for {len(item2idx)} items")

        # Build user history for negative sampling
        print("Building user history...")
        user_history = build_user_history(train_mapped, test_df=test_mapped)

        # Create negative sampler
        n_items = len(item2idx)
        neg_sampler_test = NegativeSampler(
            n_items=n_items,
            user_history=user_history,
            seed=args.seed + 1,
        )

        # Run evaluation with tie-breaking
        print("Running evaluation...")
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

        # Compare results
        old_recall = result['recall@10']
        old_precision = result['precision@10']
        old_ndcg = result['ndcg@10']

        new_recall = new_results['recall@K']
        new_precision = new_results['precision@K']
        new_ndcg = new_results['ndcg@K']

        print(f"\n📊 Results Comparison:")
        print(f"{'Metric':<15} {'Original':<12} {'With Tie-Break':<15} {'Change':<10}")
        print(f"{'-'*55}")
        print(f"{'Recall@10':<15} {old_recall:<12.6f} {new_recall:<15.6f} {new_recall-old_recall:>+9.6f}")
        print(f"{'Precision@10':<15} {old_precision:<12.6f} {new_precision:<15.6f} {new_precision-old_precision:>+9.6f}")
        print(f"{'NDCG@10':<15} {old_ndcg:<12.6f} {new_ndcg:<15.6f} {new_ndcg-old_ndcg:>+9.6f}")

        # Diagnosis
        print(f"\n🔍 Diagnosis:")
        if old_recall > 0.9 and new_recall < 0.05:
            print(f"✅ EQUAL SCORES BUG CONFIRMED!")
            print(f"   - Original recall: {old_recall:.1%} (all items ranked equally)")
            print(f"   - With tie-breaking: {new_recall:.1%} (random selection)")
            print(f"   - Expected: ~{1/(1+result['test_negatives']):.1%} for random chance")
            print(f"   → Model outputs identical scores for all items (equal scores bug)")
        elif abs(new_recall - old_recall) < 0.001:
            print(f"✓ MODEL WORKING CORRECTLY")
            print(f"   - Recall unchanged: {old_recall:.1%} → {new_recall:.1%}")
            print(f"   → Model produces discriminative scores (no equal scores bug)")
        elif old_recall > 0.9 and new_recall > 0.5:
            print(f"⚠️  UNEXPECTED: High recall persists")
            print(f"   - Possible data leakage or other issue")
        else:
            print(f"📈 PARTIAL CHANGE")
            print(f"   - Recall changed from {old_recall:.1%} to {new_recall:.1%}")
            print(f"   - Some score ties exist but not complete collapse")

        # Save updated results
        output_file = result_file.parent / f"{job_name}_results_with_tie_breaking.json"
        output_data = {
            **result,
            'original_recall@10': old_recall,
            'original_precision@10': old_precision,
            'original_ndcg@10': old_ndcg,
            'tie_break_recall@10': new_recall,
            'tie_break_precision@10': new_precision,
            'tie_break_ndcg@10': new_ndcg,
            'equal_scores_bug': old_recall > 0.9 and new_recall < 0.05,
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Saved to: {output_file}")

        return output_data

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate models with tie-breaking")

    # Data paths
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--results_dir', type=str, default='results/amazon-books',
                        help='Directory with result JSON files')

    # Column names
    parser.add_argument('--user_id_col', type=str, default='userId')
    parser.add_argument('--item_id_col', type=str, default='asin')

    # Eval params
    parser.add_argument('--seed', type=int, default=42)

    # Which models
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to evaluate (default: all)')

    args = parser.parse_args()

    print("="*80)
    print("RE-EVALUATION WITH TIE-BREAKING")
    print("="*80)

    # Load data once
    print(f"\nLoading data...")
    train_df = pd.read_parquet(args.train_path)
    test_df = pd.read_parquet(args.test_path)
    print(f"  Train: {len(train_df):,} interactions")
    print(f"  Test: {len(test_df):,} interactions")

    # Find result files
    results_dir = Path(args.results_dir)
    result_files = list(results_dir.glob('*_results.json'))

    # Filter out tie-breaking results from previous runs
    result_files = [f for f in result_files if 'tie_breaking' not in f.stem]

    if args.models:
        result_files = [f for f in result_files if any(m in f.stem for m in args.models)]

    if not result_files:
        print(f"\n❌ No result files found in {results_dir}")
        return

    print(f"\nFound {len(result_files)} models to re-evaluate:")
    for f in result_files:
        print(f"  - {f.stem}")

    # Re-evaluate each model
    summary = []
    for result_file in sorted(result_files):
        result = reevaluate_model(result_file, train_df, test_df, args)
        if result:
            summary.append(result)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if summary:
        print(f"\n{'Model':<45} {'Original R@10':<15} {'Tie-Break R@10':<15} {'Bug?':<6}")
        print("-"*85)

        for result in summary:
            name = result['job_name'].replace('amazon-books-hybrid-', '')
            orig = result['original_recall@10']
            new = result['tie_break_recall@10']
            bug = "YES" if result.get('equal_scores_bug', False) else "no"
            print(f"{name:<45} {orig:<15.6f} {new:<15.6f} {bug:<6}")

        # Count bugs
        n_bugs = sum(1 for r in summary if r.get('equal_scores_bug', False))
        print(f"\n📊 Models with equal scores bug: {n_bugs}/{len(summary)}")

        if n_bugs > 0:
            print(f"\n💡 Next steps:")
            print(f"   - These models output identical scores for all items")
            print(f"   - Root cause likely: tag format issues or training failure")
            print(f"   - See DEBUGGING_SUMMARY.md for detailed analysis")


if __name__ == "__main__":
    main()
