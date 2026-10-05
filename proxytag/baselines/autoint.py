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
from proxytag.core.data import build_id_mappings, build_user_history, NegativeSampler, InBatchTrainDataset
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
    1. User/Item ID embeddings (+ optional tag embedding as 3rd field)
    2. Stacked interacting layers (multi-head self-attention across fields)
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
        tag_embed_dim=None,  # None = no tags, else dim of pooled tag embeddings
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.tag_embed_dim = tag_embed_dim

        # ID embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # Optional tag projection
        self.item_tag_embs = None
        if tag_embed_dim is not None:
            self.tag_projection = nn.Linear(tag_embed_dim, embed_dim)

        # Interacting layers (stacked self-attention)
        self.interacting_layers = nn.ModuleList([
            InteractingLayer(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Final MLP
        # Input: flattened features from all fields
        num_fields = 3 if tag_embed_dim else 2  # user, item, [tag]
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

    def set_item_tag_embs(self, item_tag_embs):
        """Set pre-computed pooled tag embeddings [n_items, tag_embed_dim]."""
        self.item_tag_embs = item_tag_embs

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        if self.tag_embed_dim and self.item_tag_embs is not None:
            tag_raw = self.item_tag_embs[item_ids]
            tag_emb = self.tag_projection(tag_raw)
            feat_embed = torch.stack([user_emb, item_emb, tag_emb], dim=1)
        else:
            feat_embed = torch.stack([user_emb, item_emb], dim=1)

        for layer in self.interacting_layers:
            feat_embed = layer(feat_embed)

        feat_flat = feat_embed.flatten(start_dim=1)
        logits = self.mlp(feat_flat).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            return self._training_step_bpr(batch)
        return self._training_step_inbatch(batch)

    def _training_step_bpr(self, batch):
        user_ids, pos_ids, neg_ids_list = batch
        batch_size = user_ids.shape[0]
        n_neg = neg_ids_list.shape[1]

        pos_logits = self(user_ids, pos_ids)
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, n_neg)
        neg_logits = self(user_ids_expanded.flatten(), neg_ids_list.flatten())
        neg_logits = neg_logits.view(batch_size, n_neg)

        diff = pos_logits.unsqueeze(1) - neg_logits
        loss = -F.logsigmoid(diff).mean()

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _training_step_inbatch(self, batch):
        user_ids, item_ids = batch
        B = user_ids.size(0)

        user_exp = user_ids.unsqueeze(1).expand(-1, B).reshape(-1)
        item_exp = item_ids.unsqueeze(0).expand(B, -1).reshape(-1)
        all_logits = self(user_exp, item_exp)
        score_matrix = all_logits.view(B, B)

        labels = torch.arange(B, device=self.device)
        loss = F.cross_entropy(score_matrix, labels)

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, pos_ids, neg_ids_list = batch
        batch_size = user_ids.shape[0]
        n_neg = neg_ids_list.shape[1]

        pos_logits = self(user_ids, pos_ids)
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, n_neg)
        neg_logits = self(user_ids_expanded.flatten(), neg_ids_list.flatten())
        neg_logits = neg_logits.view(batch_size, n_neg)

        diff = pos_logits.unsqueeze(1) - neg_logits
        loss = -F.logsigmoid(diff).mean()

        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

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


from proxytag.baselines.eval_utils import evaluate_ranking


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
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[200, 80])
    parser.add_argument('--dropout', type=float, default=0.0)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_train_neg', type=int, default=4)
    parser.add_argument('--n_val_neg', type=int, default=50)
    parser.add_argument('--in_batch_negatives', action='store_true', default=True,
                        help='Use in-batch negative sampling for training (default: True)')
    parser.add_argument('--no_in_batch_negatives', action='store_false', dest='in_batch_negatives',
                        help='Use per-sample random negative sampling instead')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_check_interval', type=float, default=1.0)

    # Evaluation
    parser.add_argument('--n_test_neg', type=int, default=999)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help='Max interactions for test evaluation (uniformly sampled)')

    # Output
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/autoint')
    parser.add_argument('--results_dir', type=str, default='results')

    # Tags (optional)
    parser.add_argument('--tags_path', type=str, default=None,
                        help='Path to tag parquet (item_id + tags columns). Omit for no-tag baseline.')
    parser.add_argument('--max_tags', type=int, default=16,
                        help='Max tags per item to embed before pooling (default: 16)')

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
    if args.in_batch_negatives:
        train_dataset = InBatchTrainDataset(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers if args.num_workers > 0 else 0,
            persistent_workers=True if args.num_workers > 0 else False,
        )
    else:
        train_dataset = AutoIntDataset(train_df, neg_sampler_train, n_neg=args.n_train_neg)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    val_dataset = AutoIntDataset(val_df, neg_sampler_train, n_neg=args.n_val_neg)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Load tags if provided
    tag_embed_dim = None
    item_tag_embs = None
    if args.tags_path:
        from proxytag.baselines.tag_utils import load_and_pool_tags
        print(f"\n[4.5/5] Loading tag embeddings from: {args.tags_path}")
        item_tag_embs = load_and_pool_tags(
            args.tags_path, item2idx, item_id_col=args.item_id_col, max_tags=args.max_tags
        )
        tag_embed_dim = item_tag_embs.shape[1]
        print(f"  Tag embedding dim: {tag_embed_dim}")
        print(f"  Items with tags: {(item_tag_embs.abs().sum(dim=1) > 0).sum():,} / {n_items:,}")

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
        tag_embed_dim=tag_embed_dim,
    )
    if item_tag_embs is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.set_item_tag_embs(item_tag_embs.to(device))

    # Setup callbacks
    job_ckpt_dir = os.path.join(args.ckpt_dir, args.job_name)
    os.makedirs(job_ckpt_dir, exist_ok=True)
    monitor_metric = 'val/loss'
    checkpoint_callback = ModelCheckpoint(
        dirpath=job_ckpt_dir,
        filename='best',
        monitor=monitor_metric,
        mode='min',
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
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
        val_check_interval=args.val_check_interval,
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
    if item_tag_embs is not None:
        best_model.set_item_tag_embs(item_tag_embs.to(device))
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

    def score_fn(user_ids, item_ids):
        return best_model(user_ids.to(device), item_ids.to(device))

    test_metrics, bucket_df = evaluate_ranking(
        score_fn=score_fn,
        test_df=test_df,
        neg_sampler=neg_sampler_test,
        K=args.K,
        n_neg=args.n_test_neg,
        item_bucket=item_bucket,
        seed=args.seed + 2,
        max_eval_samples=args.max_eval_samples,
    )

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    for metric, value in sorted(test_metrics.items()):
        if not metric.endswith('_std') and metric != 'n_eval':
            print(f"  {metric}: {value:.4f}")
    print(f"  N eval: {test_metrics['n_eval']:,}")

    if bucket_df is not None:
        print(f"\n  By popularity bucket:")
        print(bucket_df.to_string(index=False))

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)

    epochs_trained = early_stop_callback.stopped_epoch + 1 if early_stop_callback.stopped_epoch > 0 else trainer.current_epoch + 1
    stopped_early = early_stop_callback.stopped_epoch > 0
    best_epoch = epochs_trained - args.patience if stopped_early else epochs_trained
    best_val_loss = float(checkpoint_callback.best_model_score)
    final_train_loss = float(trainer.callback_metrics.get('train/loss_epoch', 0))
    final_val_loss = float(trainer.callback_metrics.get('val/loss', 0))
    train_val_gap = final_val_loss - final_train_loss

    results = {
        'job_name': args.job_name,
        'model': 'AutoInt',
        'tags_path': args.tags_path,
        'best_checkpoint': checkpoint_callback.best_model_path,
        'best_val_loss': best_val_loss,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'train_val_gap': train_val_gap,
        'epochs_trained': epochs_trained,
        'best_epoch': best_epoch,
        'stopped_early': stopped_early,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
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
