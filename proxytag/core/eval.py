import gc
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def eval_split_fast(
    model,
    df,
    neg_sampler,
    n_items: int,
    n_neg: int = 1000,
    K: int = 10,
    max_eval_samples: int = None,
    batch_pos: int = 4,   # number of positives per batch
    seed: int = 42,
    compute_loss: bool = True,
):
    """
    LLM-REC style evaluation, fully vectorized in PyTorch.

    Evaluates per positive:
      1 positive + n_neg negatives
    """
    model.eval()
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)

    # Create torch generator for deterministic tie-breaking
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(seed)

    rows = df[["uid", "iid"]].values
    if max_eval_samples and len(rows) > max_eval_samples:
        rows = rows[rng.choice(len(rows), size=max_eval_samples, replace=False)]

    precisions, recalls, ndcgs, losses = [], [], [], []
    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    with torch.no_grad():
        for start in tqdm(range(0, len(rows), batch_pos)):
            batch = rows[start : start + batch_pos]
            B = len(batch)

            users = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            pos_items = torch.tensor(batch[:, 1], dtype=torch.long, device=device)

            # ---- sample negatives per positive ----
            neg_items = [
                neg_sampler.sample(int(u), n_neg) for u in users.tolist()
            ]
            neg_items = torch.tensor(neg_items, dtype=torch.long, device=device)  # [B, n_neg]

            # ---- candidates [B, 1+n_neg] ----
            candidates = torch.cat(
                [pos_items.unsqueeze(1), neg_items],
                dim=1,
            )

            # ---- expand users ----
            users_exp = users.unsqueeze(1).expand_as(candidates)

            # ---- flatten for model ----
            u_flat = users_exp.reshape(-1)
            i_flat = candidates.reshape(-1)

            # ---- forward once ----
            user_vec, item_vec, tags_vec = model(u_flat, i_flat)  # calls user-conditioned attention
            alpha = torch.sigmoid(model.tag_gate)         # scalar
            item_combined_vec = alpha * item_vec + (1 - alpha) * tags_vec
            scores = (user_vec * item_combined_vec).sum(dim=1)
            scores = torch.clamp(scores, -20, 20)
            scores = scores.view(B, -1)  # [B, 1+n_neg]

            # ---- loss ----
            if compute_loss:
                labels = torch.zeros_like(scores)
                labels[:, 0] = 1.0
                loss = bce(scores, labels)
                losses.append(loss.item())

            # ---- ranking metrics ----
            # Add random tie-breaking with seed to handle equal scores bug
            # When scores are equal, topk returns original order (stable sort)
            # This causes 99% recall bug - add small random noise to break ties fairly
            tie_breaker = torch.rand(scores.shape, generator=torch_rng, device=device) * 1e-8
            scores_with_ties = scores + tie_breaker
            topk_scores, topk_idx = torch.topk(scores_with_ties, K, dim=1)

            # Hit@K
            is_pos = (topk_idx == 0)            # [B, K]
            hits = is_pos.any(dim=1).float()    # [B]
            
            precisions.extend((hits / K).cpu().tolist())
            recalls.extend(hits.cpu().tolist())
            
            # NDCG@K (single positive)
            pos_rank = is_pos.float().argmax(dim=1)  # [B]
            ndcg = torch.zeros(B, device=device)
            ndcg[hits.bool()] = 1.0 / torch.log2(pos_rank[hits.bool()].float() + 2.0)
            
            ndcgs.extend(ndcg.cpu().tolist())

    results = {
        "precision@K": float(np.mean(precisions)) if precisions else 0.0,
        "recall@K": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@K": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "precision@K_std": float(np.std(precisions)) if precisions else 0.0,
        "recall@K_std": float(np.std(recalls)) if recalls else 0.0,
        "ndcg@K_std": float(np.std(ndcgs)) if ndcgs else 0.0,
        "n_eval": len(rows),
    }
    if compute_loss:
        results["loss"] = float(np.mean(losses)) if losses else 0.0

    return results



def eval_split_by_bucket(
    model,
    test_df,
    neg_sampler,
    item_bucket,              # np.ndarray [n_items] with 0..n_bins
    n_items,
    n_neg=1000,
    K=10,
    seed=42,
    max_eval_samples=None,
    debug=False,  # Add debug flag
):
    model.eval()
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)

    # Random tie-breaker generator
    tie_breaker_rng = np.random.default_rng(seed)

    # Debug: Check if tag embeddings are properly loaded
    if debug:
        print(f"\n{'='*60}")
        print(f"MODEL DEBUG INFO")
        print(f"{'='*60}")
        print(f"use_item_id: {model.hparams.use_item_id}")
        print(f"use_tags: {model.hparams.use_tags}")
        print(f"tag_encoding_type: {model.hparams.tag_encoding_type}")
        if model.item_tag_embs is not None:
            print(f"item_tag_embs: shape={model.item_tag_embs.shape}, has_nan={torch.isnan(model.item_tag_embs).any()}")
            print(f"item_tag_mask: shape={model.item_tag_mask.shape}, any_tags={model.item_tag_mask.any()}")
            print(f"Total items with tags: {model.item_tag_mask.any(dim=1).sum().item()}")
        else:
            print(f"⚠️ WARNING: item_tag_embs is None!")
        print(f"{'='*60}\n")

    rows = test_df[["uid", "iid"]].values
    if max_eval_samples and len(rows) > max_eval_samples:
        rows = rows[rng.choice(len(rows), size=max_eval_samples, replace=False)]

    # aggregate overall
    overall = {"precision": [], "recall": [], "ndcg": []}

    # aggregate by bucket
    by_bucket = defaultdict(lambda: {"precision": [], "recall": [], "ndcg": [], "n": 0})

    with torch.no_grad():
        for u, pos_i in tqdm(rows, file=sys.stdout):
            u = int(u)
            # pos_i is already mapped to integer index, don't convert
            # (Amazon ASINs would fail int() conversion anyway)

            neg_items = neg_sampler.sample(u, n_neg)

            # Debug: check negative sampling
            if debug and len(overall["precision"]) < 3:
                print(f"\n=== Negative Sampling Debug {len(overall['precision'])+1} ===")
                print(f"User {u}, pos_item {pos_i}")
                print(f"Requested {n_neg} negatives, got {len(neg_items)}")
                print(f"User history size: {len(neg_sampler.user_history.get(u, set()))}")
                print(f"Total candidates for ranking: {1 + len(neg_items)}")

            candidates = np.asarray([pos_i] + neg_items, dtype=np.int64)
            labels = np.zeros(len(candidates), dtype=np.int32)
            labels[0] = 1

            u_batch = torch.full((len(candidates),), u, dtype=torch.long, device=device)
            i_batch = torch.tensor(candidates, dtype=torch.long, device=device)

            user_vec, item_vec, tags_vec = model(u_batch, i_batch)  # calls user-conditioned attention
            alpha = torch.sigmoid(model.tag_gate)         # scalar
            item_combined_vec = alpha * item_vec + (1 - alpha) * tags_vec

            # Debug: check for NaN in intermediate values
            if debug and len(overall["precision"]) < 5:
                print(f"\n=== Debug Sample {len(overall['precision']) + 1} ===")
                print(f"User: {u}, Positive item: {pos_i}")
                print(f"Alpha (tag gate): {alpha.item():.4f}")
                print(f"user_vec: shape={user_vec.shape}, has_nan={torch.isnan(user_vec).any().item()}, mean={user_vec.mean().item():.4f}")
                print(f"item_vec: shape={item_vec.shape}, has_nan={torch.isnan(item_vec).any().item()}, mean={item_vec.mean().item():.4f}")
                print(f"tags_vec: shape={tags_vec.shape}, has_nan={torch.isnan(tags_vec).any().item()}, mean={tags_vec.mean().item():.4f}")
                print(f"item_combined_vec: has_nan={torch.isnan(item_combined_vec).any().item()}")
                if torch.isnan(tags_vec).any():
                    print(f"  ⚠️ tags_vec contains NaN! Checking tag embeddings...")
                    print(f"  item_tag_mask sum: {model.item_tag_mask[i_batch].sum().item()}")

            scores = (user_vec * item_combined_vec).sum(dim=1)
            scores = torch.clamp(scores, -20, 20).cpu().numpy()

            # Debug: print score statistics
            if debug and len(overall["precision"]) < 5:
                print(f"Scores shape: {scores.shape}")
                print(f"Positive score: {scores[0]:.6f}")
                print(f"Negative scores (first 5): {scores[1:6]}")
                print(f"Score mean: {np.nanmean(scores):.6f}, std: {np.nanstd(scores):.8f}")
                print(f"Score min: {scores.min():.6f}, max: {scores.max():.6f}")
                print(f"Unique scores: {len(np.unique(scores))}")
                print(f"All equal? {np.allclose(scores, scores[0], atol=1e-6)}")
                print(f"Has NaN: {np.isnan(scores).any()}, All NaN: {np.isnan(scores).all()}")

            # Add random tie-breaking to handle equal scores bug
            # When scores are equal, argsort returns original order (stable sort)
            # This causes 99% recall bug - add small random noise to break ties fairly
            tie_breaker = tie_breaker_rng.random(len(scores)) * 1e-8
            scores_with_ties = scores + tie_breaker
            order = np.argsort(-scores_with_ties)
            topk_labels = labels[order][:K]

            hits = int(topk_labels.sum())
            precision = hits / K
            recall = hits  # single positive → recall is 1 if hit else 0

            # NDCG@K (single positive → IDCG=1)
            ndcg = 0.0
            for r, rel in enumerate(topk_labels):
                if rel == 1:
                    ndcg = 1.0 / np.log2(r + 2)
                    break

            # overall
            overall["precision"].append(precision)
            overall["recall"].append(recall)
            overall["ndcg"].append(ndcg)

            # bucket
            b = int(item_bucket[pos_i])
            by_bucket[b]["precision"].append(precision)
            by_bucket[b]["recall"].append(recall)
            by_bucket[b]["ndcg"].append(ndcg)
            by_bucket[b]["n"] += 1

    # summarize
    results_overall = {
        "precision@K": float(np.mean(overall["precision"])) if overall["precision"] else 0.0,
        "recall@K": float(np.mean(overall["recall"])) if overall["recall"] else 0.0,
        "ndcg@K": float(np.mean(overall["ndcg"])) if overall["ndcg"] else 0.0,
        "precision@K_std": float(np.std(overall["precision"])) if overall["precision"] else 0.0,
        "recall@K_std": float(np.std(overall["recall"])) if overall["recall"] else 0.0,
        "ndcg@K_std": float(np.std(overall["ndcg"])) if overall["ndcg"] else 0.0,
        "n_eval": int(len(rows)),
    }

    # bucket table
    bucket_rows = []
    for b in sorted(by_bucket.keys()):
        vals = by_bucket[b]
        bucket_rows.append({
            "bucket": b,
            "n": vals["n"],
            "precision@K": float(np.mean(vals["precision"])) if vals["precision"] else 0.0,
            "recall@K": float(np.mean(vals["recall"])) if vals["recall"] else 0.0,
            "ndcg@K": float(np.mean(vals["ndcg"])) if vals["ndcg"] else 0.0,
            "precision@K_std": float(np.std(vals["precision"])) if vals["precision"] else 0.0,
            "recall@K_std": float(np.std(vals["recall"])) if vals["recall"] else 0.0,
            "ndcg@K_std": float(np.std(vals["ndcg"])) if vals["ndcg"] else 0.0,
        })

    bucket_df = pd.DataFrame(bucket_rows).sort_values("bucket")

    return results_overall, bucket_df
    