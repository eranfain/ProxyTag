#!/usr/bin/env python3
"""
Paired significance test between two recommendation models.

Loads two model checkpoints, computes per-user Recall@K and NDCG@K on
the test set using identical negative samples, then runs a paired t-test.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from tqdm import tqdm

from proxytag.core.data import build_id_mappings, build_user_history, NegativeSampler
from proxytag.core.tags import build_item_tag_tensors
from proxytag.core.model import HybridRecLightning
from proxytag.baselines.autoint import AutoInt
from proxytag.baselines.dcnv2 import DCNv2

MODEL_CLASSES = {
    "hybrid": HybridRecLightning,
    "autoint": AutoInt,
    "dcnv2": DCNv2,
}


def load_model(ckpt_path, model_type, device):
    """Load a model from checkpoint."""
    cls = MODEL_CLASSES[model_type]
    model = cls.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model = model.to(device)
    return model


@torch.no_grad()
def score_batch(model, model_type, user_ids, item_ids):
    """Score (user, item) pairs. Returns 1-D tensor of scores."""
    if model_type == "hybrid":
        user_vec, item_vec, tags_vec = model(user_ids, item_ids)
        alpha = torch.sigmoid(model.tag_gate)
        item_combined = alpha * item_vec + (1 - alpha) * tags_vec
        scores = (user_vec * item_combined).sum(dim=1)
        scores = torch.clamp(scores, -20, 20)
    else:
        scores = model(user_ids, item_ids)
    return scores


@torch.no_grad()
def compute_per_user_metrics(model, model_type, user_positives, neg_sampler,
                             n_neg, K, device, seed):
    """
    Compute per-user Recall@K and NDCG@K.

    Returns dict: user_id -> {"recall": float, "ndcg": float}
    """
    rng = np.random.default_rng(seed)
    results = {}

    for user_id, pos_items in tqdm(user_positives.items(), desc="Evaluating",
                                   file=sys.stdout):
        user_recalls = []
        user_ndcgs = []

        for pos_item in pos_items:
            neg_items = neg_sampler.sample(user_id, n_neg)

            # candidates: [pos_item] + negatives
            candidates = [pos_item] + neg_items
            u_tensor = torch.full((len(candidates),), user_id,
                                  dtype=torch.long, device=device)
            i_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

            scores = score_batch(model, model_type, u_tensor, i_tensor)
            scores = scores.cpu().numpy()

            # Tie-breaking
            tie_breaker = rng.random(len(scores)) * 1e-8
            order = np.argsort(-(scores + tie_breaker))

            # pos_item is at index 0 in candidates
            pos_rank_in_topk = np.where(order[:K] == 0)[0]
            hit = len(pos_rank_in_topk) > 0

            user_recalls.append(1.0 if hit else 0.0)
            if hit:
                rank = pos_rank_in_topk[0]  # 0-indexed position in top-K
                user_ndcgs.append(1.0 / np.log2(rank + 2))
            else:
                user_ndcgs.append(0.0)

        results[user_id] = {
            "recall": float(np.mean(user_recalls)),
            "ndcg": float(np.mean(user_ndcgs)),
        }

    return results


def cohens_d(a, b):
    """Compute Cohen's d for paired samples."""
    diff = a - b
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def main():
    parser = argparse.ArgumentParser(
        description="Paired significance test between two models")

    # Model A
    parser.add_argument("--model_a_ckpt", type=str, required=True)
    parser.add_argument("--model_a_type", type=str, required=True,
                        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--model_a_name", type=str, default="Model A")

    # Model B
    parser.add_argument("--model_b_ckpt", type=str, required=True)
    parser.add_argument("--model_b_type", type=str, required=True,
                        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--model_b_name", type=str, default="Model B")

    # Data paths
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--user_id_col", type=str, default="userId")
    parser.add_argument("--item_id_col", type=str, default="movieId")

    # Tag-related (only needed if a model is hybrid)
    parser.add_argument("--items_data_path", type=str, default=None,
                        help="Path to items/tags parquet (required if a model is hybrid)")
    parser.add_argument("--sentence_transformers_model", type=str,
                        default="all-MiniLM-L6-v2")
    parser.add_argument("--max_tags", type=int, default=24)

    # Evaluation
    parser.add_argument("--max_users", type=int, default=1000)
    parser.add_argument("--n_test_neg", type=int, default=999)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_path", type=str, default=None,
                        help="Save results JSON to this path")

    args = parser.parse_args()

    needs_tags = args.model_a_type == "hybrid" or args.model_b_type == "hybrid"
    if needs_tags and not args.items_data_path:
        parser.error("--items_data_path is required when a model is hybrid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load data and build mappings ----
    print("=" * 70)
    print("PAIRED SIGNIFICANCE TEST")
    print("=" * 70)
    print(f"  {args.model_a_name}: {args.model_a_ckpt} ({args.model_a_type})")
    print(f"  {args.model_b_name}: {args.model_b_ckpt} ({args.model_b_type})")
    print(f"  K={args.K}, n_neg={args.n_test_neg}, max_users={args.max_users}")
    print()

    print("[1/4] Loading data and building ID mappings...")
    train_df = pd.read_parquet(args.train_path)
    val_df = pd.read_parquet(args.val_path)
    test_df = pd.read_parquet(args.test_path)

    train_df, val_df, test_df, user2idx, idx2user, item2idx, idx2item = \
        build_id_mappings(train_df, val_df, test_df,
                          user_id_col=args.user_id_col,
                          item_id_col=args.item_id_col)

    n_users = len(user2idx)
    n_items = len(item2idx)
    user_history = build_user_history(train_df, val_df, test_df)

    print(f"  Users: {n_users:,}  Items: {n_items:,}  Test rows: {len(test_df):,}")

    # ---- Sample users ----
    rng = np.random.default_rng(args.seed)
    unique_test_users = test_df["uid"].unique()
    if len(unique_test_users) > args.max_users:
        sampled_users = rng.choice(unique_test_users, size=args.max_users,
                                   replace=False)
    else:
        sampled_users = unique_test_users
    sampled_users = set(sampled_users.tolist())

    # Build per-user positive items (only for sampled users)
    user_positives = defaultdict(list)
    for _, row in test_df.iterrows():
        uid = int(row["uid"])
        if uid in sampled_users:
            user_positives[uid].append(int(row["iid"]))

    print(f"  Sampled users: {len(user_positives):,}")
    total_positives = sum(len(v) for v in user_positives.values())
    print(f"  Total test positives: {total_positives:,}")

    # ---- Load tag tensors if needed ----
    if needs_tags:
        print("\n  Loading tag tensors...")
        item_tag_embs, item_tag_mask = build_item_tag_tensors(
            items_df_path=args.items_data_path,
            item2idx=item2idx,
            model_name=args.sentence_transformers_model,
            max_tags=args.max_tags,
            device="cpu",
            normalize=True,
            overwrite=False,
            item_id_col=args.item_id_col,
        )

    # ---- Load models ----
    print("\n[2/4] Loading models...")
    model_a = load_model(args.model_a_ckpt, args.model_a_type, device)
    if args.model_a_type == "hybrid":
        model_a.set_item_tag_tensors(item_tag_embs, item_tag_mask)
        model_a = model_a.to(device)

    model_b = load_model(args.model_b_ckpt, args.model_b_type, device)
    if args.model_b_type == "hybrid":
        model_b.set_item_tag_tensors(item_tag_embs, item_tag_mask)
        model_b = model_b.to(device)

    # ---- Evaluate both models (same negatives via same seed) ----
    print(f"\n[3/4] Evaluating {args.model_a_name}...")
    neg_sampler_a = NegativeSampler(n_items, user_history, seed=args.seed + 10)
    results_a = compute_per_user_metrics(
        model_a, args.model_a_type, user_positives, neg_sampler_a,
        args.n_test_neg, args.K, device, seed=args.seed)

    print(f"\n[3/4] Evaluating {args.model_b_name}...")
    neg_sampler_b = NegativeSampler(n_items, user_history, seed=args.seed + 10)
    results_b = compute_per_user_metrics(
        model_b, args.model_b_type, user_positives, neg_sampler_b,
        args.n_test_neg, args.K, device, seed=args.seed)

    # ---- Paired t-test ----
    print(f"\n[4/4] Running paired t-test...")
    common_users = sorted(set(results_a.keys()) & set(results_b.keys()))

    recall_a = np.array([results_a[u]["recall"] for u in common_users])
    recall_b = np.array([results_b[u]["recall"] for u in common_users])
    ndcg_a = np.array([results_a[u]["ndcg"] for u in common_users])
    ndcg_b = np.array([results_b[u]["ndcg"] for u in common_users])

    recall_t, recall_p = ttest_rel(recall_a, recall_b)
    ndcg_t, ndcg_p = ttest_rel(ndcg_a, ndcg_b)

    recall_d = cohens_d(recall_a, recall_b)
    ndcg_d = cohens_d(ndcg_a, ndcg_b)

    # ---- Report ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Paired users: {len(common_users):,}")
    print()

    header = f"{'Metric':<15} {'':>5} {args.model_a_name:>18} {args.model_b_name:>18} {'t-stat':>10} {'p-value':>10} {'Cohen d':>10} {'Sig':>5}"
    print(header)
    print("-" * len(header))

    def sig_marker(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    print(f"{'Recall@'+str(args.K):<15} {'mean':>5} {recall_a.mean():>18.4f} {recall_b.mean():>18.4f} {recall_t:>10.3f} {recall_p:>10.6f} {recall_d:>10.3f} {sig_marker(recall_p):>5}")
    print(f"{'':.<15} {'std':>5} {recall_a.std():>18.4f} {recall_b.std():>18.4f}")
    print()
    print(f"{'NDCG@'+str(args.K):<15} {'mean':>5} {ndcg_a.mean():>18.4f} {ndcg_b.mean():>18.4f} {ndcg_t:>10.3f} {ndcg_p:>10.6f} {ndcg_d:>10.3f} {sig_marker(ndcg_p):>5}")
    print(f"{'':.<15} {'std':>5} {ndcg_a.std():>18.4f} {ndcg_b.std():>18.4f}")
    print()

    print("Significance: * p<0.05  ** p<0.01  *** p<0.001")
    print(f"Positive t-stat / Cohen's d => {args.model_a_name} is better")
    print("=" * 70)

    # ---- Save results ----
    output = {
        "model_a": {"name": args.model_a_name, "ckpt": args.model_a_ckpt, "type": args.model_a_type},
        "model_b": {"name": args.model_b_name, "ckpt": args.model_b_ckpt, "type": args.model_b_type},
        "config": {"K": args.K, "n_test_neg": args.n_test_neg, "max_users": args.max_users, "seed": args.seed},
        "n_paired_users": len(common_users),
        "recall": {
            "mean_a": float(recall_a.mean()), "std_a": float(recall_a.std()),
            "mean_b": float(recall_b.mean()), "std_b": float(recall_b.std()),
            "t_stat": float(recall_t), "p_value": float(recall_p),
            "cohens_d": float(recall_d),
        },
        "ndcg": {
            "mean_a": float(ndcg_a.mean()), "std_a": float(ndcg_a.std()),
            "mean_b": float(ndcg_b.mean()), "std_b": float(ndcg_b.std()),
            "t_stat": float(ndcg_t), "p_value": float(ndcg_p),
            "cohens_d": float(ndcg_d),
        },
    }

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
