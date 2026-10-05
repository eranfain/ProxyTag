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
if not hasattr(np.__config__, 'get_info'):
    np.__config__.get_info = lambda x: {}
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


def get_cf_item_emb(train_df, item2idx, user_id_col, item_id_col, embedding_dim=64, cf_method='bpr'):
    """
    Train CF model and extract L2-normalized item embeddings.

    Args:
        train_df: Training interactions DataFrame
        item2idx: Mapping from item ID to index
        user_id_col: User ID column name
        item_id_col: Item ID column name
        embedding_dim: Latent factor dimension
        cf_method: 'als' or 'bpr'

    Returns:
        cf_emb: Dict mapping item ID to normalized embedding vector
    """
    from scipy.sparse import csr_matrix

    print("--- Building Interaction Matrix ---")

    user_map = {u: i for i, u in enumerate(train_df[user_id_col].unique())}

    train_df['uid'] = train_df[user_id_col].map(user_map)
    train_df['iid'] = train_df[item_id_col].map(item2idx)
    train_df['rating'] = 1

    print(f"Training CF model ({cf_method.upper()})...")
    train_mat = csr_matrix((train_df['rating'], (train_df['uid'], train_df['iid'])),
                           shape=(len(user_map), len(item2idx)))

    if cf_method == 'als':
        model = implicit.als.AlternatingLeastSquares(
            factors=embedding_dim, regularization=0.1, iterations=20
        )
        model.fit(train_mat)
        item_embeddings = model.item_factors
    elif cf_method == 'bpr':
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=embedding_dim, regularization=0.01, iterations=100
        )
        model.fit(train_mat)
        # BPR adds an extra bias column — strip it for item-item similarity
        item_embeddings = model.item_factors[:, :-1]
    else:
        raise ValueError(f"Unknown cf_method: {cf_method}")

    cf_emb = {}
    for mid, idx in item2idx.items():
        emb = item_embeddings[idx]
        norm = np.linalg.norm(emb)
        cf_emb[mid] = emb / norm if norm > 0 else emb

    print(f"Completed CF training ({cf_method.upper()}, {item_embeddings.shape[1]} dims)")

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


def compute_tag_embeddings(items_df, item_col, tag_col, model_name, max_tags=16, device=None):
    """
    Compute tag embeddings using Sentence Transformers.

    Uses cached embeddings from the parquet if available (written by
    build_item_tag_tensors), otherwise encodes from scratch.

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
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Computing Tag Embeddings (device={device}) ---")

    item_to_idx = {iid: i for i, iid in enumerate(items_df[item_col].values)}
    n_items = len(items_df)

    # Try loading from parquet cache (written by build_item_tag_tensors)
    if 'embs' in items_df.columns and 'emb_mask' in items_df.columns:
        print("  Loading from cached embeddings in parquet...")
        emb_dim = None
        for v in items_df['embs'].values:
            if v is not None and len(v) > 0:
                emb_dim = len(v[0])
                break
        if emb_dim is not None:
            item_tag_vectors = np.zeros((n_items, emb_dim), dtype=np.float32)
            for i, row in enumerate(items_df.itertuples()):
                row_embs = row.embs
                if row_embs is None or len(row_embs) == 0:
                    continue
                arr = np.stack([np.asarray(e, dtype=np.float32) for e in row_embs[:max_tags]])
                item_tag_vectors[i] = arr.mean(axis=0)

            norms = np.linalg.norm(item_tag_vectors, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            item_tag_vectors = item_tag_vectors / norms

            n_with_tags = (np.linalg.norm(item_tag_vectors, axis=1) > 0).sum()
            print(f"  Items: {n_items:,}, with tags: {n_with_tags:,} ({n_with_tags/n_items:.2%})")
            return item_tag_vectors, item_to_idx

    # Fall back to encoding from scratch
    model = SentenceTransformer(model_name, device=device)
    emb_dim = model.get_sentence_embedding_dimension()
    item_tag_vectors = np.zeros((n_items, emb_dim), dtype=np.float32)

    print(f"  Items: {n_items:,}")
    print(f"  Embedding dim: {emb_dim}")

    for i, row in tqdm(items_df.iterrows(), total=len(items_df), desc="Encoding tags"):
        tags_str = row[tag_col]

        if pd.isna(tags_str) or not tags_str:
            continue

        # Split tags: prefer pipe, fall back to comma
        if '|' in tags_str:
            tags = [t.strip() for t in tags_str.split('|') if t.strip()][:max_tags]
        else:
            tags = [t.strip() for t in tags_str.split(',') if t.strip()][:max_tags]

        if not tags:
            continue

        tag_embs = model.encode(tags, convert_to_numpy=True, show_progress_bar=False)
        item_tag_vectors[i] = tag_embs.mean(axis=0)

    # Normalize
    norms = np.linalg.norm(item_tag_vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    item_tag_vectors = item_tag_vectors / norms

    n_with_tags = (np.linalg.norm(item_tag_vectors, axis=1) > 0).sum()
    print(f"  Items with tags: {n_with_tags:,} ({n_with_tags/n_items:.2%})")

    return item_tag_vectors, item_to_idx


def build_cooccurrence_matrix(train_df, user_id_col, item_id_col, item2idx, panel_indices, min_cooccur=3):
    """
    Build item co-occurrence counts restricted to panel items.

    For each user, counts how many times each pair of their items (that are
    both in the panel) co-occur. Returns a dense [num_panel, num_panel] matrix.

    Args:
        train_df: Training interactions DataFrame
        user_id_col: User ID column name
        item_id_col: Item ID column name
        item2idx: Mapping from item ID to global index
        panel_indices: Array of global item indices in the panel
        min_cooccur: Minimum co-occurrence count to keep (below is zeroed out)

    Returns:
        cooccur: [num_panel, num_panel] co-occurrence count matrix
    """
    print("--- Building Co-occurrence Matrix ---")

    panel_set = set(panel_indices.tolist())
    global_to_panel = {g: p for p, g in enumerate(panel_indices)}
    n_panel = len(panel_indices)

    idx2item = {v: k for k, v in item2idx.items()}

    user_groups = train_df.groupby(user_id_col)[item_id_col].apply(list)

    cooccur = np.zeros((n_panel, n_panel), dtype=np.int32)

    for items in tqdm(user_groups, desc="Building co-occurrence"):
        panel_local = []
        for item_id in items:
            if item_id in item2idx:
                gidx = item2idx[item_id]
                if gidx in panel_set:
                    panel_local.append(global_to_panel[gidx])
        if len(panel_local) < 2:
            continue
        for i in range(len(panel_local)):
            for j in range(i + 1, len(panel_local)):
                cooccur[panel_local[i], panel_local[j]] += 1
                cooccur[panel_local[j], panel_local[i]] += 1

    cooccur[cooccur < min_cooccur] = 0
    np.fill_diagonal(cooccur, 0)

    n_nonzero = np.count_nonzero(cooccur) // 2
    print(f"  Panel size: {n_panel}")
    print(f"  Co-occurring pairs (>= {min_cooccur}): {n_nonzero:,}")
    avg_neighbors = (cooccur > 0).sum(axis=1).mean()
    print(f"  Avg neighbors per item: {avg_neighbors:.1f}")

    return cooccur


def compute_rnok(panel_item_idx, panel_cf_vectors, panel_tag_embeddings, cooccur_row, K=50):
    """
    Compute RNO@K for a single panel item.

    Finds behaviorally-related items (from co-occurrence) that CF misses,
    then checks how many of those the tag embeddings recover.

    Args:
        panel_item_idx: Index within panel
        panel_cf_vectors: [num_panel, cf_dim] CF embeddings
        panel_tag_embeddings: [num_panel, tag_dim] tag embeddings
        cooccur_row: [num_panel] co-occurrence counts for this item
        K: Number of top neighbors

    Returns:
        rnok: float or None if no CF blind spots exist
    """
    gt_neighbors = set(np.where(cooccur_row > 0)[0])
    if not gt_neighbors:
        return None

    cf_vec = panel_cf_vectors[panel_item_idx]
    cf_sims = panel_cf_vectors @ cf_vec
    cf_sims[panel_item_idx] = -np.inf
    K_eff = min(K, len(panel_cf_vectors) - 1)
    cf_top_k = set(np.argsort(-cf_sims)[:K_eff].tolist())

    blind_spots = gt_neighbors - cf_top_k
    if not blind_spots:
        return None

    tag_vec = panel_tag_embeddings[panel_item_idx]
    tag_sims = panel_tag_embeddings @ tag_vec
    tag_sims[panel_item_idx] = -np.inf
    tag_top_k = set(np.argsort(-tag_sims)[:K_eff].tolist())

    recovered = len(tag_top_k & blind_spots)
    return recovered / min(K_eff, len(blind_spots))


def compute_r2_proxy(panel_cf_vectors, panel_tag_embeddings, user_item_matrix, panel_indices, n_sample_pairs=50000, seed=42):
    """
    R²-based proxy: measures how much tag similarity adds to CF similarity
    in predicting user-space item similarity.

    For sampled item pairs, computes:
      - user_sim: cosine similarity of user interaction vectors (ground truth)
      - cf_sim: cosine similarity of CF embeddings
      - tag_sim: cosine similarity of tag embeddings

    Fits linear regression: user_sim ~ cf_sim + tag_sim
    Returns R²(cf+tags), R²(cf_only), and the incremental R².

    Args:
        panel_cf_vectors: [num_panel, cf_dim] L2-normalized CF embeddings
        panel_tag_embeddings: [num_panel, tag_dim] L2-normalized tag embeddings
        user_item_matrix: [num_users, num_items] sparse interaction matrix
        panel_indices: global item indices corresponding to panel rows
        n_sample_pairs: number of item pairs to sample
        seed: random seed

    Returns:
        dict with r2_combined, r2_cf_only, r2_incremental, beta_cf, beta_tag
    """
    from sklearn.linear_model import LinearRegression
    from scipy.sparse import issparse

    rng = np.random.default_rng(seed)
    n_panel = len(panel_indices)

    # Build user vectors for panel items (sparse columns → dense)
    panel_user_vectors = user_item_matrix[:, panel_indices].T  # [n_panel, n_users]
    if issparse(panel_user_vectors):
        panel_user_vectors = panel_user_vectors.toarray()
    panel_user_vectors = panel_user_vectors.astype(np.float32)

    # L2-normalize user vectors
    norms = np.linalg.norm(panel_user_vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    panel_user_vectors = panel_user_vectors / norms

    # Sample pairs
    n_pairs = min(n_sample_pairs, n_panel * (n_panel - 1) // 2)
    idx_i = rng.integers(0, n_panel, size=n_pairs * 2)
    idx_j = rng.integers(0, n_panel, size=n_pairs * 2)
    # Remove self-pairs and deduplicate
    mask = idx_i != idx_j
    idx_i = idx_i[mask][:n_pairs]
    idx_j = idx_j[mask][:n_pairs]

    # Compute similarities for sampled pairs
    user_sims = np.sum(panel_user_vectors[idx_i] * panel_user_vectors[idx_j], axis=1)
    cf_sims = np.sum(panel_cf_vectors[idx_i] * panel_cf_vectors[idx_j], axis=1)
    tag_sims = np.sum(panel_tag_embeddings[idx_i] * panel_tag_embeddings[idx_j], axis=1)

    # Linear regression: user_sim ~ cf_sim + tag_sim
    X_combined = np.column_stack([cf_sims, tag_sims])
    X_cf_only = cf_sims.reshape(-1, 1)
    y = user_sims

    reg_combined = LinearRegression().fit(X_combined, y)
    reg_cf_only = LinearRegression().fit(X_cf_only, y)

    r2_combined = reg_combined.score(X_combined, y)
    r2_cf_only = reg_cf_only.score(X_cf_only, y)
    r2_incremental = r2_combined - r2_cf_only

    return {
        'r2_combined': float(r2_combined),
        'r2_cf_only': float(r2_cf_only),
        'r2_incremental': float(r2_incremental),
        'beta_cf': float(reg_combined.coef_[0]),
        'beta_tag': float(reg_combined.coef_[1]),
    }


def compute_dcg_comp(panel_item_idx, panel_cf_vectors, panel_tag_embeddings, cooccur_row):
    """
    Discounted Complementarity Gain.

    Measures how well tags surface co-occurring items that CF misses.
    Each co-occurring item is weighted by its CF rank percentile (0→1),
    so items CF ranks well contribute near zero. The tag side uses
    inverse log-rank (NDCG-style) to reward items tags rank highly.

    Score = mean(cf_percentile_j / log2(tag_rank_j + 1))
            for all co-occurring items j.

    High score = tags confidently rank items that CF misses.
    Low score  = tags only rank items CF already captures.

    Args:
        panel_item_idx: Index within panel
        panel_cf_vectors: [num_panel, cf_dim] CF embeddings
        panel_tag_embeddings: [num_panel, tag_dim] tag embeddings
        cooccur_row: [num_panel] co-occurrence counts for this item

    Returns:
        dcg_comp: float or None if no co-occurring items
    """
    cooccur_items = np.where(cooccur_row > 0)[0]
    if len(cooccur_items) == 0:
        return None

    n_panel = len(panel_cf_vectors)

    cf_vec = panel_cf_vectors[panel_item_idx]
    cf_sims = panel_cf_vectors @ cf_vec
    cf_sims[panel_item_idx] = -np.inf

    tag_vec = panel_tag_embeddings[panel_item_idx]
    tag_sims = panel_tag_embeddings @ tag_vec
    tag_sims[panel_item_idx] = -np.inf

    # CF ranks (1-indexed) and percentile (0→1)
    cf_rank_order = np.argsort(-cf_sims)
    cf_ranks = np.empty_like(cf_rank_order)
    cf_ranks[cf_rank_order] = np.arange(len(cf_rank_order)) + 1

    # Tag ranks (1-indexed)
    tag_rank_order = np.argsort(-tag_sims)
    tag_ranks = np.empty_like(tag_rank_order)
    tag_ranks[tag_rank_order] = np.arange(len(tag_rank_order)) + 1

    cf_pct = cf_ranks[cooccur_items].astype(np.float64) / (n_panel - 1)
    tag_r = tag_ranks[cooccur_items].astype(np.float64)
    scores = cf_pct / np.log2(tag_r + 1)

    return float(np.mean(scores))


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
    n_bins=10,
    cooccur_matrix=None,
    cooccur_panel_indices=None,
    user_item_matrix=None,
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
        cooccur_matrix: Optional [num_panel, num_panel] co-occurrence matrix
            indexed by cooccur_panel_indices. When provided, also computes RNO@K.
        cooccur_panel_indices: Global item indices corresponding to cooccur_matrix
            rows/columns. Required when cooccur_matrix is provided.
        user_item_matrix: Optional sparse [num_users, num_items] interaction matrix.
            When provided, computes R²-based proxy metric.

    Returns:
        results: Dict with overall and per-bin NO@K (and RNO@K) metrics
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

    compute_rnok_flag = cooccur_matrix is not None

    # Re-index cooccurrence matrix to match filtered panel
    filtered_cooccur = None
    if compute_rnok_flag:
        cooccur_idx_map = {int(gidx): pos for pos, gidx in enumerate(cooccur_panel_indices)}
        filtered_to_full = [cooccur_idx_map[int(gidx)] for gidx in panel_indices]
        filtered_cooccur = cooccur_matrix[np.ix_(filtered_to_full, filtered_to_full)]

    # Initialize storage
    recalls = []
    bin_recalls = {i: [] for i in range(n_bins)}
    rnok_values = []
    bin_rnok = {i: [] for i in range(n_bins)}
    dcg_comp_values = []
    bin_dcg_comp = {i: [] for i in range(n_bins)}

    # Evaluate each panel item
    for i, item_id in enumerate(tqdm(panel_ids, desc="Computing NO@K")):
        # Compute NO@K within panel
        recall = compute_similarity_correlation_panel(
            i, panel_cf_embeddings, panel_tag_embeddings, K=K
        )

        recalls.append(recall)
        bin_idx = item_bins.get(item_id, 0)
        bin_recalls[bin_idx].append(recall)

        # Compute DCG_comp (requires co-occurrence matrix)
        if compute_rnok_flag:
            dcg_val = compute_dcg_comp(
                i, panel_cf_embeddings, panel_tag_embeddings,
                filtered_cooccur[i]
            )
            if dcg_val is not None:
                dcg_comp_values.append(dcg_val)
                bin_dcg_comp[bin_idx].append(dcg_val)

        # Compute RNO@K if co-occurrence matrix provided
        if compute_rnok_flag:
            rnok_val = compute_rnok(
                i, panel_cf_embeddings, panel_tag_embeddings,
                filtered_cooccur[i], K=K
            )
            if rnok_val is not None:
                rnok_values.append(rnok_val)
                bin_rnok[bin_idx].append(rnok_val)

    # Compute R² proxy (panel-level, not per-item)
    if user_item_matrix is not None:
        r2_results = compute_r2_proxy(
            panel_cf_embeddings, panel_tag_embeddings,
            user_item_matrix, panel_indices,
        )
    else:
        r2_results = None

    # Aggregate results
    results = {
        'overall': {
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'n_items': len(recalls),
        },
    }

    if r2_results is not None:
        results['r2_proxy'] = r2_results

    if dcg_comp_values:
        results['dcg_comp_overall'] = {
            'mean_dcg_comp': np.mean(dcg_comp_values),
            'std_dcg_comp': np.std(dcg_comp_values),
            'n_items': len(dcg_comp_values),
        }

    if compute_rnok_flag and rnok_values:
        results['rnok_overall'] = {
            'mean_rnok': np.mean(rnok_values),
            'std_rnok': np.std(rnok_values),
            'n_items': len(rnok_values),
        }

    # Per-bin results
    for bin_idx in range(n_bins):
        if bin_recalls[bin_idx]:
            results[f'bin_{bin_idx}'] = {
                'mean_recall': np.mean(bin_recalls[bin_idx]),
                'n_items': len(bin_recalls[bin_idx]),
            }
        if bin_dcg_comp[bin_idx]:
            results[f'dcg_comp_bin_{bin_idx}'] = {
                'mean_dcg_comp': np.mean(bin_dcg_comp[bin_idx]),
                'n_items': len(bin_dcg_comp[bin_idx]),
            }
        if compute_rnok_flag and bin_rnok[bin_idx]:
            results[f'rnok_bin_{bin_idx}'] = {
                'mean_rnok': np.mean(bin_rnok[bin_idx]),
                'n_items': len(bin_rnok[bin_idx]),
            }

    # Print summary
    print(f"\n--- Results Summary ---")
    print(f"Overall NO@{K}: {results['overall']['mean_recall']:.4f} ± {results['overall']['std_recall']:.4f}")
    if 'r2_proxy' in results:
        r = results['r2_proxy']
        print(f"R² proxy: combined={r['r2_combined']:.4f}, cf_only={r['r2_cf_only']:.4f}, incremental={r['r2_incremental']:.6f} (β_tag={r['beta_tag']:.4f})")
    if 'dcg_comp_overall' in results:
        r = results['dcg_comp_overall']
        print(f"Overall DCG_comp: {r['mean_dcg_comp']:.4f} ± {r['std_dcg_comp']:.4f} ({r['n_items']} items with co-occurring tag neighbors)")
    print(f"Items evaluated: {results['overall']['n_items']}")

    if compute_rnok_flag and rnok_values:
        r = results['rnok_overall']
        print(f"Overall RNO@{K}: {r['mean_rnok']:.4f} ± {r['std_rnok']:.4f} ({r['n_items']} items with blind spots)")

    print(f"\nBy Popularity Bin:")
    for bin_idx in range(n_bins):
        key = f'bin_{bin_idx}'
        if key in results:
            r = results[key]
            extra = ""
            dcg_key = f'dcg_comp_bin_{bin_idx}'
            if dcg_key in results:
                extra += f", DCG_comp={results[dcg_key]['mean_dcg_comp']:.4f}"
            rnok_key = f'rnok_bin_{bin_idx}'
            if rnok_key in results:
                extra += f", RNO@{K}={results[rnok_key]['mean_rnok']:.4f}"
            print(f"  Bin {bin_idx}: NO@{K}={r['mean_recall']:.4f}{extra}, n={r['n_items']}")

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
    parser.add_argument('--max_tags', type=int, default=16)

    # Similarity computation
    parser.add_argument('--K', type=int, default=10,
                        help='Number of neighbors for NO@K (default: 10)')
    parser.add_argument('--use_interaction_cf', action='store_true',
                        help='Use interaction-based CF similarity instead of learned embeddings (recommended)')
    parser.add_argument('--cf_method', type=str, default='bpr', choices=['als', 'bpr'],
                        help='CF model for proxy metric (default: bpr)')

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
        print(f"\n--- Using {args.cf_method.upper()}-based CF ---")
        cf_emb_dict = get_cf_item_emb(
            train_df,
            item2idx,
            args.user_id_col,
            args.item_id_col,
            cf_method=args.cf_method,
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
