#!/usr/bin/env python3
"""
AutoInt Baseline

Based on: "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
- Uses multi-head self-attention to automatically learn feature interactions
- Combines with deep neural network
- No need for manual feature engineering
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

# Project imports
from proxytag.core.data import build_id_mappings, build_user_history, NegativeSampler
from proxytag.core.utils import build_item_popularity_percentiles


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for feature interaction."""
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Linear projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: [B, num_fields, embed_dim]
        Returns:
            output: [B, num_fields, embed_dim]
        """
        batch_size, num_fields, _ = x.shape

        # Linear projections and split into heads
        # [B, num_fields, embed_dim] -> [B, num_fields, n_heads, head_dim] -> [B, n_heads, num_fields, head_dim]
        Q = self.W_q(x).view(batch_size, num_fields, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, num_fields, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, num_fields, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, n_heads, num_fields, num_fields]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values: [B, n_heads, num_fields, head_dim]
        attn_out = torch.matmul(attn_probs, V)

        # Concatenate heads: [B, num_fields, embed_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, num_fields, self.embed_dim)

        # Output projection
        output = self.W_o(attn_out)

        return output


class InteractingLayer(nn.Module):
    """Interacting layer with self-attention and residual connection."""
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, num_fields, embed_dim]
        Returns:
            output: [B, num_fields, embed_dim]
        """
        # Self-attention with residual connection
        attn_out = self.self_attention(x)
        attn_out = self.dropout(attn_out)
        output = self.layer_norm(x + attn_out)  # Residual connection

        return output


class AutoInt(pl.LightningModule):
    """
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks.

    Architecture:
    1. User/Item ID embeddings
    2. Stacked interacting layers (multi-head self-attention)
    3. Final MLP for prediction
    """
    def __init__(
        self,
        n_users,
        n_items,
        embed_dim=32,
        n_heads=4,
        n_layers=3,
        mlp_dims=[200, 80],
        dropout=0.0,
        lr=1e-4,
        weight_decay=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        # ID embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # Interacting layers (stacked self-attention)
        self.interacting_layers = nn.ModuleList([
            InteractingLayer(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Final MLP
        # Input: flattened features from all fields
        num_fields = 2  # user, item
        mlp_input_dim = num_fields * embed_dim

        mlp_layers = []
        prev_dim = mlp_input_dim

        for hidden_dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        """
        Args:
            user_ids: [B]
            item_ids: [B]
        Returns:
            logits: [B]
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [B, embed_dim]
        item_emb = self.item_embedding(item_ids)  # [B, embed_dim]

        # Stack fields: [B, num_fields, embed_dim]
        feat_embed = torch.stack([user_emb, item_emb], dim=1)

        # Apply interacting layers (self-attention)
        for layer in self.interacting_layers:
            feat_embed = layer(feat_embed)

        # Flatten for MLP: [B, num_fields * embed_dim]
        feat_flat = feat_embed.flatten(start_dim=1)

        # Final prediction
        logits = self.mlp(feat_flat).squeeze(-1)  # [B]

        return logits

    def training_step(self, batch, batch_idx):
        user_ids, pos_ids, neg_ids_list = batch
        batch_size = user_ids.shape[0]
        n_neg = neg_ids_list.shape[1]

        # Positive scores
        pos_logits = self(user_ids, pos_ids)  # [B]

        # Negative scores
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, n_neg)  # [B, n_neg]
        neg_logits = self(user_ids_expanded.flatten(), neg_ids_list.flatten())  # [B*n_neg]
        neg_logits = neg_logits.view(batch_size, n_neg)  # [B, n_neg]

        # BPR loss
        pos_logits_expanded = pos_logits.unsqueeze(1)  # [B, 1]
        diff = pos_logits_expanded - neg_logits  # [B, n_neg]
        loss = -F.logsigmoid(diff).mean()

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


class AutoIntDataset(Dataset):
    """Dataset for AutoInt training with negative sampling."""
    def __init__(self, df, neg_sampler, n_neg=4):
        self.user_ids = torch.from_numpy(df['uid'].values).long()
        self.item_ids = torch.from_numpy(df['iid'].values).long()
        self.neg_sampler = neg_sampler
        self.n_neg = n_neg

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        pos_item = self.item_ids[idx]

        # Sample negatives
        neg_items = self.neg_sampler.sample(user_id.item(), self.n_neg)
        neg_items = torch.tensor(neg_items, dtype=torch.long)

        return user_id, pos_item, neg_items


def compute_metrics(ranks, K):
    """Compute ranking metrics from ranks."""
    ranks = np.array(ranks)
    precision = (ranks <= K).mean()
    recall = (ranks <= K).mean()
    dcg = (1.0 / np.log2(ranks + 1)) * (ranks <= K)
    idcg = 1.0 / np.log2(2)
    ndcg = (dcg / idcg).mean()

    return {
        f'precision@{K}': precision,
        f'recall@{K}': recall,
        f'ndcg@{K}': ndcg,
    }


@torch.no_grad()
def evaluate_test_set(model, test_df, neg_sampler, K, n_neg, batch_size, device,
                      item_bucket=None, seed=42, max_eval_samples=None):
    """Evaluate model on test set with ranking metrics and optional bucket breakdown."""
    model.eval()

    rng = np.random.default_rng(seed)
    tie_breaker_rng = np.random.default_rng(seed)

    if max_eval_samples and len(test_df) > max_eval_samples:
        idx = rng.choice(len(test_df), size=max_eval_samples, replace=False)
        test_df = test_df.iloc[idx].reset_index(drop=True)

    test_dataset = AutoIntDataset(test_df, neg_sampler, n_neg=n_neg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    overall = {"precision": [], "recall": [], "ndcg": []}
    by_bucket = defaultdict(lambda: {"precision": [], "recall": [], "ndcg": [], "n": 0})

    for batch in tqdm(test_loader, desc='Test evaluation', file=sys.stdout):
        user_ids, pos_ids, neg_ids_list = batch
        user_ids = user_ids.to(device)
        pos_ids = pos_ids.to(device)
        neg_ids_list = neg_ids_list.to(device)

        batch_size_actual = user_ids.shape[0]
        n_neg_actual = neg_ids_list.shape[1]

        # Score positive items
        pos_scores = model(user_ids, pos_ids)  # [B]

        # Score negative items
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, n_neg_actual)
        neg_scores = model(user_ids_expanded.flatten(), neg_ids_list.flatten())
        neg_scores = neg_scores.view(batch_size_actual, n_neg_actual)  # [B, n_neg]

        # Per-example metrics with tie-breaking
        for i in range(batch_size_actual):
            pos_score = pos_scores[i].item()
            neg_score_list = neg_scores[i].cpu().numpy()

            scores = np.concatenate([[pos_score], neg_score_list])
            tie_breaker = tie_breaker_rng.random(len(scores)) * 1e-8
            order = np.argsort(-(scores + tie_breaker))

            labels = np.zeros(len(scores), dtype=np.int32)
            labels[0] = 1
            topk_labels = labels[order][:K]

            hits = int(topk_labels.sum())
            precision = hits / K
            recall = float(hits)

            ndcg = 0.0
            for r, rel in enumerate(topk_labels):
                if rel == 1:
                    ndcg = 1.0 / np.log2(r + 2)
                    break

            overall["precision"].append(precision)
            overall["recall"].append(recall)
            overall["ndcg"].append(ndcg)

            if item_bucket is not None:
                b = int(item_bucket[pos_ids[i].item()])
                by_bucket[b]["precision"].append(precision)
                by_bucket[b]["recall"].append(recall)
                by_bucket[b]["ndcg"].append(ndcg)
                by_bucket[b]["n"] += 1

    results = {
        "precision@K": float(np.mean(overall["precision"])) if overall["precision"] else 0.0,
        "recall@K": float(np.mean(overall["recall"])) if overall["recall"] else 0.0,
        "ndcg@K": float(np.mean(overall["ndcg"])) if overall["ndcg"] else 0.0,
        "precision@K_std": float(np.std(overall["precision"])) if overall["precision"] else 0.0,
        "recall@K_std": float(np.std(overall["recall"])) if overall["recall"] else 0.0,
        "ndcg@K_std": float(np.std(overall["ndcg"])) if overall["ndcg"] else 0.0,
        "n_eval": len(test_df),
    }

    bucket_df = None
    if item_bucket is not None:
        bucket_rows = []
        for b in sorted(by_bucket.keys()):
            vals = by_bucket[b]
            bucket_rows.append({
                "bucket": b, "n": vals["n"],
                "precision@K": float(np.mean(vals["precision"])) if vals["precision"] else 0.0,
                "recall@K": float(np.mean(vals["recall"])) if vals["recall"] else 0.0,
                "ndcg@K": float(np.mean(vals["ndcg"])) if vals["ndcg"] else 0.0,
                "precision@K_std": float(np.std(vals["precision"])) if vals["precision"] else 0.0,
                "recall@K_std": float(np.std(vals["recall"])) if vals["recall"] else 0.0,
                "ndcg@K_std": float(np.std(vals["ndcg"])) if vals["ndcg"] else 0.0,
            })
        bucket_df = pd.DataFrame(bucket_rows).sort_values("bucket")

    return results, bucket_df


def main():
    parser = argparse.ArgumentParser(description="Train AutoInt baseline")

    # Data paths
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)

    # Column names
    parser.add_argument('--user_id_col', type=str, default='userId')
    parser.add_argument('--item_id_col', type=str, default='movieId')

    # Model hyperparameters
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[200, 80])
    parser.add_argument('--dropout', type=float, default=0.0)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_train_neg', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)

    # Evaluation
    parser.add_argument('--n_test_neg', type=int, default=99)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help='Max interactions for test evaluation (uniformly sampled)')

    # Output
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/autoint')
    parser.add_argument('--results_dir', type=str, default='results')

    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print("=" * 80)
    print("AUTOINT BASELINE")
    print("=" * 80)
    print(f"Job: {args.job_name}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Attention heads: {args.n_heads}")
    print(f"Attention layers: {args.n_layers}")
    print(f"MLP dims: {args.mlp_dims}")
    print("=" * 80)

    # Set seed
    pl.seed_everything(args.seed)

    # Load data
    print("\n[1/5] Loading data...")
    train_df = pd.read_parquet(args.train_path)
    val_df = pd.read_parquet(args.val_path)
    test_df = pd.read_parquet(args.test_path)

    print(f"  Train: {len(train_df):,}")
    print(f"  Val: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")

    # Build ID mappings
    print("\n[2/5] Building ID mappings...")
    train_df, val_df, test_df, user2idx, idx2user, item2idx, idx2item = build_id_mappings(
        train_df, val_df, test_df,
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col
    )

    n_users = len(user2idx)
    n_items = len(item2idx)

    print(f"  Users: {n_users:,}")
    print(f"  Items: {n_items:,}")

    # Build user history
    print("\n[3/5] Building user history...")
    all_interactions = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user_history = build_user_history(all_interactions)

    # Create negative samplers
    neg_sampler_train = NegativeSampler(
        user_history=user_history,
        n_items=n_items,
        seed=args.seed,
    )

    # Create datasets
    print("\n[4/5] Creating datasets...")
    train_dataset = AutoIntDataset(train_df, neg_sampler_train, n_neg=args.n_train_neg)
    val_dataset = AutoIntDataset(val_df, neg_sampler_train, n_neg=args.n_train_neg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Create model
    print("\n[5/5] Training model...")
    model = AutoInt(
        n_users=n_users,
        n_items=n_items,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_dims=args.mlp_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Setup callbacks
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='best',
        monitor='train/loss_epoch',
        mode='min',
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor='train/loss_epoch',
        patience=args.patience,
        mode='min',
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        precision='bf16-mixed',
        gradient_clip_val=5.0,
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # Load best checkpoint for test evaluation
    print("\n[TEST EVALUATION] Loading best checkpoint...")
    best_model = AutoInt.load_from_checkpoint(checkpoint_callback.best_model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = best_model.to(device)
    best_model.eval()

    # Create negative sampler for test
    neg_sampler_test = NegativeSampler(
        user_history=user_history,
        n_items=n_items,
        seed=args.seed + 1,
    )

    # Build popularity buckets
    item_count, item_bucket = build_item_popularity_percentiles(train_df, n_items, n_bins=10)

    # Evaluate on test set
    print(f"[TEST EVALUATION] Evaluating with K={args.K}, n_test_neg={args.n_test_neg}...")
    test_metrics, bucket_df = evaluate_test_set(
        model=best_model,
        test_df=test_df,
        neg_sampler=neg_sampler_test,
        K=args.K,
        n_neg=args.n_test_neg,
        batch_size=args.batch_size,
        device=device,
        item_bucket=item_bucket,
        seed=args.seed + 2,
        max_eval_samples=args.max_eval_samples,
    )

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"  Precision@{args.K}: {test_metrics['precision@K']:.4f}")
    print(f"  Recall@{args.K}: {test_metrics['recall@K']:.4f}")
    print(f"  NDCG@{args.K}: {test_metrics['ndcg@K']:.4f}")
    print(f"  N eval: {test_metrics['n_eval']:,}")

    if bucket_df is not None:
        print(f"\n  By popularity bucket:")
        print(bucket_df.to_string(index=False))

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    results = {
        'job_name': args.job_name,
        'best_checkpoint': checkpoint_callback.best_model_path,
        'best_train_loss': float(checkpoint_callback.best_model_score),
        'test_metrics': test_metrics,
        'hyperparameters': vars(args),
    }

    results_file = os.path.join(args.results_dir, f'{args.job_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    if bucket_df is not None:
        buckets_file = os.path.join(args.results_dir, f'{args.job_name}_buckets_eval.csv')
        bucket_df.to_csv(buckets_file, index=False)
        print(f"Bucket results saved to: {buckets_file}")

    print(f"\nResults saved to: {results_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
