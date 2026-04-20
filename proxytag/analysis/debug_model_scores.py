#!/usr/bin/env python3
"""
Debug script to understand why some models produce equal scores (~99% recall).

Loads trained models and analyzes their learned parameters and score distributions.
"""

import argparse
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Import from project
from proxytag.core.model import HybridRecLightning
from proxytag.core.data import build_id_mappings, build_user_history, NegativeSampler
from proxytag.core.tags import build_item_tag_tensors


def load_model_checkpoint(result_file):
    """Load model from results file metadata."""
    with open(result_file, 'r') as f:
        result = json.load(f)

    job_name = result['job_name']
    checkpoint_path = f"checkpoints/{job_name}_best.ckpt"

    if not Path(checkpoint_path).exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return None, result

    # Load model
    model = HybridRecLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    return model, result


def analyze_model_parameters(model, name):
    """Analyze learned parameter statistics."""
    print(f"\n{'='*80}")
    print(f"PARAMETER ANALYSIS: {name}")
    print(f"{'='*80}")

    # User embeddings
    user_emb = model.user_emb.weight.data
    print(f"\nUser embeddings:")
    print(f"  Shape: {user_emb.shape}")
    print(f"  Mean: {user_emb.mean():.4f}, Std: {user_emb.std():.4f}")
    print(f"  Min: {user_emb.min():.4f}, Max: {user_emb.max():.4f}")

    # Item embeddings
    item_emb = model.item_emb.weight.data
    print(f"\nItem embeddings:")
    print(f"  Shape: {item_emb.shape}")
    print(f"  Mean: {item_emb.mean():.4f}, Std: {item_emb.std():.4f}")
    print(f"  Min: {item_emb.min():.4f}, Max: {item_emb.max():.4f}")

    # Tag gate (hybrid gating parameter)
    tag_gate = model.tag_gate.data.item()
    alpha = torch.sigmoid(torch.tensor(tag_gate)).item()
    print(f"\nTag gate:")
    print(f"  Raw value: {tag_gate:.4f}")
    print(f"  Alpha (sigmoid): {alpha:.4f}")
    print(f"  Item weight: {alpha:.1%}, Tag weight: {1-alpha:.1%}")

    # Tag projection layer
    if hasattr(model, 'tag_proj'):
        tag_proj_weight = model.tag_proj.weight.data
        print(f"\nTag projection layer:")
        print(f"  Shape: {tag_proj_weight.shape}")
        print(f"  Mean: {tag_proj_weight.mean():.4f}, Std: {tag_proj_weight.std():.4f}")

    # Tag attention
    if hasattr(model, 'tag_attn'):
        # Multi-head attention has in_proj_weight and out_proj
        in_proj = model.tag_attn.in_proj_weight.data
        out_proj = model.tag_attn.out_proj.weight.data
        print(f"\nTag attention:")
        print(f"  In projection - Mean: {in_proj.mean():.4f}, Std: {in_proj.std():.4f}")
        print(f"  Out projection - Mean: {out_proj.mean():.4f}, Std: {out_proj.std():.4f}")


def compute_score_statistics(model, result):
    """Compute score statistics on a small sample."""
    # This would require loading data and running inference
    # For now, just report what we have from results

    precision = result['precision@10']
    recall = result['recall@10']
    ndcg = result['ndcg@10']

    print(f"\nEvaluation metrics:")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10: {recall:.4f}")
    print(f"  NDCG@10: {ndcg:.4f}")

    # Check for equal scores bug
    if recall > 0.9:
        print(f"  ⚠️  EQUAL SCORES BUG DETECTED!")
        print(f"  Precision ≈ {precision:.4f} = Recall/10 = {recall/10:.4f}")
    elif precision > 0.02 and precision < 0.05:
        print(f"  ✓ Healthy score distribution")


def main():
    parser = argparse.ArgumentParser(description="Debug trained models")
    parser.add_argument('--results_dir', type=str, default='results/amazon-books',
                        help='Directory with result JSON files')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                        help='Specific model names to analyze (e.g., kar llm_rec_orig)')

    args = parser.parse_args()

    # Find all result files
    results_dir = Path(args.results_dir)
    result_files = list(results_dir.glob('*_results.json'))

    if args.names:
        # Filter to specific names
        result_files = [f for f in result_files if any(name in f.stem for name in args.names)]

    if not result_files:
        print(f"❌ No result files found in {results_dir}")
        return

    print(f"Found {len(result_files)} models to analyze")

    # Analyze each model
    for result_file in sorted(result_files):
        name = result_file.stem.replace('_results', '')

        model, result = load_model_checkpoint(result_file)

        if model is None:
            print(f"\n⚠️  Skipping {name} - checkpoint not found")
            compute_score_statistics(None, result)
            continue

        analyze_model_parameters(model, name)
        compute_score_statistics(model, result)


if __name__ == "__main__":
    main()
