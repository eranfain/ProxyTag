"""
Shared evaluation for all baseline models.

Single source of truth for ranking metrics: NDCG@K and Recall@K.
"""

import sys
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm


def evaluate_ranking(
    score_fn,
    test_df,
    neg_sampler,
    K=10,
    n_neg=999,
    item_bucket=None,
    train_df=None,
    n_items=None,
    n_bins=10,
    seed=42,
    max_eval_samples=None,
    batch_size=512,
):
    """
    Evaluate a model using sampled negative ranking.

    For each test interaction (user, positive_item), samples n_neg negatives,
    ranks the positive among them, and computes NDCG@K and Recall@K.

    Args:
        score_fn: Callable(user_ids: LongTensor[B], item_ids: LongTensor[B]) -> FloatTensor[B]
            Model-agnostic scoring function. Returns one score per (user, item) pair.
        test_df: DataFrame with 'uid' and 'iid' columns (contiguous indices)
        neg_sampler: NegativeSampler instance
        K: Cutoff for metrics (default 10)
        n_neg: Number of negative samples per positive (default 999)
        item_bucket: Optional np.ndarray [n_items] mapping item idx to popularity bucket
        seed: Random seed for reproducibility
        max_eval_samples: Limit evaluation to this many interactions (uniform sample)
        batch_size: Number of test interactions to process at once

    Returns:
        results: dict with recall@K, ndcg@K, precision@K (+ _std variants), n_eval
        bucket_df: DataFrame with per-bucket metrics (None if item_bucket not provided)
    """
    # Auto-compute bucket analysis if train_df provided
    if item_bucket is None and train_df is not None and n_items is not None:
        from proxytag.core.utils import build_item_popularity_percentiles
        _, item_bucket = build_item_popularity_percentiles(train_df, n_items, n_bins)

    rng = np.random.default_rng(seed)
    tie_breaker_rng = np.random.default_rng(seed)

    rows = test_df[["uid", "iid"]].values
    if max_eval_samples and len(rows) > max_eval_samples:
        idx = rng.choice(len(rows), size=max_eval_samples, replace=False)
        rows = rows[idx]

    overall = {"precision": [], "recall": [], "ndcg": []}
    by_bucket = defaultdict(lambda: {"precision": [], "recall": [], "ndcg": [], "n": 0})

    n_candidates = n_neg + 1

    for start in tqdm(range(0, len(rows), batch_size), desc="Evaluating", file=sys.stdout):
        batch = rows[start:start + batch_size]
        B = len(batch)

        # Build all candidates for the batch at once
        all_user_ids = np.empty(B * n_candidates, dtype=np.int64)
        all_item_ids = np.empty(B * n_candidates, dtype=np.int64)

        for j, (u, pos_i) in enumerate(batch):
            u = int(u)
            pos_i = int(pos_i)
            neg_items = neg_sampler.sample(u, n_neg)
            offset = j * n_candidates
            all_user_ids[offset] = u
            all_item_ids[offset] = pos_i
            all_user_ids[offset + 1:offset + n_candidates] = u
            all_item_ids[offset + 1:offset + n_candidates] = neg_items

        # Score entire batch in one call
        with torch.no_grad():
            all_scores = score_fn(
                torch.tensor(all_user_ids, dtype=torch.long),
                torch.tensor(all_item_ids, dtype=torch.long),
            ).cpu().numpy()

        # Reshape to (B, n_candidates) and compute metrics vectorized
        scores_matrix = all_scores.reshape(B, n_candidates)
        tie_breakers = tie_breaker_rng.random((B, n_candidates)) * 1e-8
        scores_matrix = scores_matrix + tie_breakers

        # Rank of positive item (index 0 in each row)
        pos_scores = scores_matrix[:, 0]
        ranks = (scores_matrix > pos_scores[:, None]).sum(axis=1)  # 0-based rank

        # Metrics
        hits = (ranks < K).astype(np.float64)
        precisions = hits / K
        recalls = hits
        ndcgs = np.where(hits > 0, 1.0 / np.log2(ranks + 2), 0.0)

        overall["precision"].extend(precisions.tolist())
        overall["recall"].extend(recalls.tolist())
        overall["ndcg"].extend(ndcgs.tolist())

        if item_bucket is not None:
            for j in range(B):
                pos_i = int(batch[j, 1])
                b = int(item_bucket[pos_i])
                by_bucket[b]["precision"].append(precisions[j])
                by_bucket[b]["recall"].append(recalls[j])
                by_bucket[b]["ndcg"].append(ndcgs[j])
                by_bucket[b]["n"] += 1

    results = {
        f"recall@{K}": float(np.mean(overall["recall"])) if overall["recall"] else 0.0,
        f"ndcg@{K}": float(np.mean(overall["ndcg"])) if overall["ndcg"] else 0.0,
        f"precision@{K}": float(np.mean(overall["precision"])) if overall["precision"] else 0.0,
        f"recall@{K}_std": float(np.std(overall["recall"])) if overall["recall"] else 0.0,
        f"ndcg@{K}_std": float(np.std(overall["ndcg"])) if overall["ndcg"] else 0.0,
        f"precision@{K}_std": float(np.std(overall["precision"])) if overall["precision"] else 0.0,
        "n_eval": len(rows),
    }

    bucket_df = None
    if item_bucket is not None:
        bucket_rows = []
        for b in sorted(by_bucket.keys()):
            vals = by_bucket[b]
            bucket_rows.append({
                "bucket": b,
                "n": vals["n"],
                f"recall@{K}": float(np.mean(vals["recall"])) if vals["recall"] else 0.0,
                f"ndcg@{K}": float(np.mean(vals["ndcg"])) if vals["ndcg"] else 0.0,
                f"precision@{K}": float(np.mean(vals["precision"])) if vals["precision"] else 0.0,
            })
        bucket_df = pd.DataFrame(bucket_rows).sort_values("bucket")

    return results, bucket_df
