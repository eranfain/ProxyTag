#!/usr/bin/env python3
"""
Proxy Metric for Tag Quality Evaluation.

Evaluates tag quality by measuring alignment between CF-based and tag-based
item similarities. This provides a fast proxy for predicting downstream
hybrid model performance without expensive training.

Algorithm:
1. Sample panel items (stratified by popularity)
2. Extract tags for panel items using different prompts
3. Compute tag embeddings (Sentence BERT)
4. Train CF model on full data
5. Extract CF item embeddings
6. For each panel item:
   - Find K most similar items by CF
   - Find K most similar items by tags
   - Compute NO@K metric
7. Aggregate NO@K (overall + by popularity bucket)
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import implicit
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from proxytag.core.data import load_df, build_id_mappings, build_user_history, NegativeSampler, RecDataModule
from proxytag.core.tags import build_item_tag_tensors
from proxytag.core.model import HybridRecLightning
from proxytag.core.utils import build_item_popularity_percentiles


def sample_panel_items(train_df, item_col, panel_size=0.05, n_bins=10, seed=42):
    """
    Sample panel items stratified by popularity.

    Args:
        train_df: Training interactions
        item_col: Item ID column name
        panel_size: Fraction of items to sample
        n_bins: Number of popularity bins for stratification
        seed: Random seed

    Returns:
        panel_items: Array of sampled item IDs
        item_bins: Dict mapping item ID to popularity bin
    """
    print("--- Sampling Panel Items ---")

    # Compute item frequencies
    item_counts = train_df[item_col].value_counts()
    n_items = len(item_counts)

    # Assign items to popularity bins
    item_bins = {}
    for i, (item_id, count) in enumerate(item_counts.items()):
        bin_idx = min(i * n_bins // n_items, n_bins - 1)
        item_bins[item_id] = bin_idx

    # Stratified sampling
    np.random.seed(seed)
    panel_items = []

    for bin_idx in range(n_bins):
        bin_items = [item for item, b in item_bins.items() if b == bin_idx]
        n_sample = max(1, int(len(bin_items) * panel_size))
        sampled = np.random.choice(bin_items, size=n_sample, replace=False)
        panel_items.extend(sampled)

    print(f"  Total items: {n_items:,}")
    print(f"  Panel items: {len(panel_items):,} ({len(panel_items)/n_items:.2%})")
    print(f"  Bins: {n_bins}")

    return np.array(panel_items), item_bins


def get_cf_item_emb(train_df, item2idx, user_id_col, item_id_col, embedding_dim=64):
    """
    Train ALS model and extract L2-normalized item embeddings.

    Args:
        train_df: Training interactions DataFrame
        item2idx: Mapping from item ID to index
        user_id_col: User ID column name
        item_id_col: Item ID column name
        embedding_dim: ALS latent factor dimension

    Returns:
        cf_emb: Dict mapping item ID to normalized embedding vector
    """
    from scipy.sparse import csr_matrix

    print("--- Building Interaction Matrix ---")

    # Get unique users
    user_map = {u: i for i, u in enumerate(train_df[user_id_col].unique())}

    train_df['uid'] = train_df[user_id_col].map(user_map)
    train_df['iid'] = train_df[item_id_col].map(item2idx)
    train_df['rating'] = 1

    # Rebuild train_mat here
    print("Start training the CF model")
    train_mat = csr_matrix((train_df['rating'], (train_df['uid'], train_df['iid'])),
                           shape=(len(user_map), len(item2idx)))

    # TRAIN THE MODEL AFTER THIS STEP
    model = implicit.als.AlternatingLeastSquares(
        factors=embedding_dim, regularization=0.1, iterations=20
    )
    model.fit(train_mat)
    item_embeddings = model.item_factors

    # map back to movieIds
    cf_emb = {mid: item_embeddings[idx] / np.linalg.norm(item_embeddings[idx])
              for mid, idx in item2idx.items()}

    print("Completed CF training")

    return cf_emb


def extract_cf_embeddings(checkpoint_path, num_items, device='cuda'):
    """
    Extract CF item embeddings from trained model.

    DEPRECATED for proxy metric: These embeddings were trained for user-item
    prediction, not item-item similarity. Use build_interaction_matrix instead
    to get true collaborative patterns.

    Args:
        checkpoint_path: Path to trained model checkpoint
        num_items: Total number of items
        device: Device to load model on

    Returns:
        item_embeddings: [num_items, hidden_dim] tensor
    """
    print("--- Extracting CF Embeddings ---")

    # Load model
    model = HybridRecLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)

    # Extract item embeddings
    with torch.no_grad():
        item_embeddings = model.item_emb.weight.cpu().numpy()

    # Don't normalize! Model was trained with raw dot products (magnitude matters)
    # Tag embeddings use cosine similarity, but NO@K only cares about ranking
    # order, not absolute scale, so different scales are fine.

    print(f"  Shape: {item_embeddings.shape}")
    print(f"  Mean norm: {np.linalg.norm(item_embeddings, axis=1).mean():.4f}")

    return item_embeddings


def compute_tag_embeddings(items_df, item_col, tag_col, model_name, max_tags=16, device='cpu'):
    """
    Compute tag embeddings using Sentence Transformers.

    Args:
        items_df: DataFrame with item IDs and tags
        item_col: Item ID column name
        tag_col: Tags column name (pipe-delimited)
        model_name: Sentence Transformer model name
        max_tags: Maximum tags per item
        device: Device for encoding

    Returns:
        item_tag_vectors: [num_items, emb_dim] averaged tag embeddings
        item_to_idx: Mapping from item ID to index
    """
    print("--- Computing Tag Embeddings ---")

    model = SentenceTransformer(model_name, device=device)
    item_to_idx = {iid: i for i, iid in enumerate(items_df[item_col].values)}

    n_items = len(items_df)
    emb_dim = model.get_sentence_embedding_dimension()
    item_tag_vectors = np.zeros((n_items, emb_dim), dtype=np.float32)

    print(f"  Items: {n_items:,}")
    print(f"  Embedding dim: {emb_dim}")

    for i, row in tqdm(items_df.iterrows(), total=len(items_df), desc="Encoding tags"):
        tags_str = row[tag_col]

        if pd.isna(tags_str) or not tags_str:
            continue

        # Split tags
        tags = [t.strip() for t in tags_str.split('|') if t.strip()][:max_tags]

        if not tags:
            continue

        # Encode tags
        tag_embs = model.encode(tags, convert_to_numpy=True, show_progress_bar=False)

        # Average pooling
        item_tag_vectors[i] = tag_embs.mean(axis=0)

    # Normalize
    norms = np.linalg.norm(item_tag_vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    item_tag_vectors = item_tag_vectors / norms

    n_with_tags = (np.linalg.norm(item_tag_vectors, axis=1) > 0).sum()
    print(f"  Items with tags: {n_with_tags:,} ({n_with_tags/n_items:.2%})")

    return item_tag_vectors, item_to_idx


def compute_similarity_correlation_panel(
    panel_item_idx,
    panel_cf_vectors,
    panel_tag_embeddings,
    K=50,
    exclude_self=True
):
    """
    - Find top-K CF neighbors
    - Compute recall (overlap) between top-K CF and top-K tag neighbors

    Args:
        panel_item_idx: Index within panel (not global index)
        panel_cf_vectors: [num_panel_items, num_features] CF representations (interaction or embeddings)
        panel_tag_embeddings: [num_panel_items, tag_dim] tag embeddings for panel
        K: Number of top similar items to compare
        exclude_self: Exclude the item itself from comparison

    Returns:
        recall: Overlap between top-K CF and top-K tag neighbors (NO@K)
    """
    # Compute similarities within panel
    cf_vec = panel_cf_vectors[panel_item_idx]
    tag_vec = panel_tag_embeddings[panel_item_idx]

    # For CF: use dot product (works for both interaction vectors and embeddings)
    cf_sims = panel_cf_vectors @ cf_vec
    # For tags: already normalized, so this is cosine similarity
    tag_sims = panel_tag_embeddings @ tag_vec

    # Exclude self if requested
    if exclude_self:
        cf_sims[panel_item_idx] = -np.inf
        tag_sims[panel_item_idx] = -np.inf

    # Cap K at available items
    K = min(K, len(panel_cf_vectors) - 1)  # -1 for excluding self

    # Get top-K by CF (sorted by similarity, descending)
    cf_top_k_idx = np.argsort(-cf_sims)[:K]

    # Get top-K by tags (for recall computation)
    tag_top_k_idx = np.argsort(-tag_sims)[:K]

    # Compute recall: overlap between top-K sets
    recall = len(set(cf_top_k_idx) & set(tag_top_k_idx)) / K

    return recall

def evaluate_tag_quality(
    panel_items,
    item_bins,
    cf_embeddings,
    tag_embeddings,
    item2idx,
    K=50,
    n_bins=10
):
    """
    Evaluate tag quality across panel items.

    Args:
        panel_items: Array of panel item IDs
        item_bins: Dict mapping item ID to popularity bin
        cf_embeddings: [num_items, cf_dim] CF embeddings
        tag_embeddings: [num_items, tag_dim] tag embeddings
        item2idx: Mapping from item ID to index
        K: Number of neighbors for NO@K
        n_bins: Number of popularity bins

    Returns:
        results: Dict with overall and per-bin NO@K metrics
    """
    print("--- Evaluating Tag Quality ---")

    # Map panel items to indices
    panel_indices = []
    panel_ids = []
    for item_id in panel_items:
        if item_id in item2idx:
            idx = item2idx[item_id]
            # Only include items with tags
            if np.linalg.norm(tag_embeddings[idx]) > 1e-6:
                panel_indices.append(idx)
                panel_ids.append(item_id)

    panel_indices = np.array(panel_indices)
    print(f"  Panel items with tags: {len(panel_indices):,}")

    # Extract panel embeddings only
    panel_cf_embeddings = cf_embeddings[panel_indices]
    panel_tag_embeddings = tag_embeddings[panel_indices]

    # Initialize storage
    recalls = []
    bin_recalls = {i: [] for i in range(n_bins)}

    # Evaluate each panel item
    for i, item_id in enumerate(tqdm(panel_ids, desc="Computing NO@K")):
        # Compute NO@K within panel
        recall = compute_similarity_correlation_panel(
            i, panel_cf_embeddings, panel_tag_embeddings, K=K
        )

        recalls.append(recall)

        # By bin
        bin_idx = item_bins.get(item_id, 0)
        bin_recalls[bin_idx].append(recall)

    # Aggregate results
    results = {
        'overall': {
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'n_items': len(recalls),
        }
    }

    # Per-bin results
    for bin_idx in range(n_bins):
        if bin_recalls[bin_idx]:
            results[f'bin_{bin_idx}'] = {
                'mean_recall': np.mean(bin_recalls[bin_idx]),
                'n_items': len(bin_recalls[bin_idx]),
            }

    # Print summary
    print(f"\n--- Results Summary ---")
    print(f"Overall NO@{K}: {results['overall']['mean_recall']:.4f} ± {results['overall']['std_recall']:.4f}")
    print(f"Items evaluated: {results['overall']['n_items']}")

    print(f"\nBy Popularity Bin:")
    for bin_idx in range(n_bins):
        key = f'bin_{bin_idx}'
        if key in results:
            r = results[key]
            print(f"  Bin {bin_idx}: NO@{K}={r['mean_recall']:.4f}, n={r['n_items']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate tag quality via proxy metric")

    # Data paths
    parser.add_argument('--train_path', type=str, required=True,
                        help='Training interactions (e.g., data/amazon-books/interactions/train.parquet)')
    parser.add_argument('--items_data_path', type=str, required=True,
                        help='Panel tag file (e.g., data/amazon-books/tags/panel/tags_multi.parquet)')
    parser.add_argument('--cf_checkpoint', type=str, default=None,
                        help='Path to trained CF model checkpoint (only needed if --use_interaction_cf is not set)')

    # Column names
    parser.add_argument('--user_id_col', type=str, default='userId')
    parser.add_argument('--item_id_col', type=str, default='movieId')
    parser.add_argument('--tag_col', type=str, default='tags')

    # Panel sampling
    parser.add_argument('--panel_items_path', type=str, default=None,
                        help='Path to pre-created panel_items.parquet (if None, will sample)')
    parser.add_argument('--panel_size', type=float, default=0.05,
                        help='Fraction of items to sample (default: 0.05)')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of popularity bins (default: 10)')

    # Tag embeddings
    parser.add_argument('--sentence_transformers_model', type=str,
                        default='all-MiniLM-L6-v2')
    parser.add_argument('--max_tags', type=int, default=24)

    # Similarity computation
    parser.add_argument('--K', type=int, default=10,
                        help='Number of neighbors for NO@K (default: 10)')
    parser.add_argument('--use_interaction_cf', action='store_true',
                        help='Use interaction-based CF similarity instead of learned embeddings (recommended)')

    # Output
    parser.add_argument('--output_path', type=str, default='proxy_results.json')

    # Infrastructure
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("="*80)
    print("PROXY METRIC EVALUATION")
    print("="*80)
    print()

    # Load training data
    train_df = load_df(args.train_path)
    print(f"Training interactions: {len(train_df):,}")

    # Load or sample panel items
    if args.panel_items_path:
        print(f"\n--- Loading Pre-created Panel ---")
        panel_df = pd.read_parquet(args.panel_items_path)
        panel_items = panel_df[args.item_id_col].values

        # Assign bins based on popularity in training data
        item_counts = train_df[args.item_id_col].value_counts()
        item_bins = {}
        for i, (item_id, count) in enumerate(item_counts.items()):
            bin_idx = min(i * args.n_bins // len(item_counts), args.n_bins - 1)
            item_bins[item_id] = bin_idx

        print(f"  Panel items: {len(panel_items):,}")
    else:
        print(f"\n--- Sampling Panel Items ---")
        panel_items, item_bins = sample_panel_items(
            train_df,
            args.item_id_col,
            panel_size=args.panel_size,
            n_bins=args.n_bins,
            seed=args.seed
        )
        print(f"  Panel items: {len(panel_items):,}")

    # Load items with tags
    print(f"\nLoading items from: {args.items_data_path}")
    items_df = pd.read_parquet(args.items_data_path)
    print(f"  Total items: {len(items_df):,}")

    # Build ID mappings (needed for CF checkpoint)
    # Create dummy val/test for compatibility
    val_df = train_df.head(100)
    test_df = train_df.head(100)

    train_df_mapped, _, _, user2idx, idx2user, item2idx, idx2item = build_id_mappings(
        train_df, val_df, test_df,
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col
    )
    num_items = len(item2idx)

    # Choose CF representation based on flag
    if args.use_interaction_cf:
        print("\n--- Using ALS-based CF ---")
        cf_emb_dict = get_cf_item_emb(
            train_df,
            item2idx,
            args.user_id_col,
            args.item_id_col
        )

        # Build dense embedding matrix from dict (already L2-normalized in get_cf_item_emb)
        emb_dim = next(iter(cf_emb_dict.values())).shape[0]
        cf_embeddings = np.zeros((num_items, emb_dim), dtype=np.float32)
        for item_id, emb in cf_emb_dict.items():
            if item_id in item2idx:
                cf_embeddings[item2idx[item_id]] = emb

        print(f"  CF representation: ALS item factors (normalized)")
        print(f"  Shape: {cf_embeddings.shape}")
        n_filled = (np.linalg.norm(cf_embeddings, axis=1) > 0).sum()
        print(f"  Items with embeddings: {n_filled:,} / {num_items:,}")

    # Compute tag embeddings
    tag_embeddings, tag_item2idx = compute_tag_embeddings(
        items_df,
        args.item_id_col,
        args.tag_col,
        args.sentence_transformers_model,
        args.max_tags,
        device=args.device
    )

    # Align indices (tag_item2idx to item2idx)
    print("\n--- Aligning Embeddings ---")
    cf_dim = cf_embeddings.shape[1]
    tag_dim = tag_embeddings.shape[1]
    print(f"  CF dim: {cf_dim}, Tag dim: {tag_dim}")

    # Dimensions may differ between CF and tag spaces — similarities are computed
    # independently per space, so this is fine (see paper Section 3.2).
    if cf_dim != tag_dim:
        print(f"  Note: CF dim ({cf_dim}) != Tag dim ({tag_dim}) — this is expected."
              f" Similarities are computed independently per space.")

    # Align tag embeddings to CF item ordering
    # For interaction-based CF, dimensions differ so we create a separate array
    aligned_tag_embeddings = np.zeros((num_items, tag_dim), dtype=np.float32)

    for item_id, tag_idx in tag_item2idx.items():
        if item_id in item2idx:
            cf_idx = item2idx[item_id]
            aligned_tag_embeddings[cf_idx] = tag_embeddings[tag_idx]

    n_aligned = (np.linalg.norm(aligned_tag_embeddings, axis=1) > 0).sum()
    print(f"  Items with aligned embeddings: {n_aligned:,} ({n_aligned/num_items:.2%})")

    # Evaluate tag quality
    results = evaluate_tag_quality(
        panel_items,
        item_bins,
        cf_embeddings,
        aligned_tag_embeddings,
        item2idx,
        K=args.K,
        n_bins=args.n_bins
    )

    # Add metadata
    results['config'] = {
        'items_data_path': args.items_data_path,
        'cf_checkpoint': args.cf_checkpoint,
        'panel_size': args.panel_size,
        'n_bins': args.n_bins,
        'K': args.K,
        'sentence_transformers_model': args.sentence_transformers_model,
    }

    # Save results
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
