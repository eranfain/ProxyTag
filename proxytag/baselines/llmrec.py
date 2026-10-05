#!/usr/bin/env python3
"""
LLM-Rec Baseline

Based on: "LLM-Rec: Personalized Recommendation via Prompting Large Language Models"
Items are represented solely by sentence-transformer encodings of LLM-enriched text (tags).
No item ID embeddings — pure text-based recommendation.

Architecture:
  - User embeddings: learned [n_users, hidden_dim]
  - Item representations: frozen tag embeddings → linear projection [tag_dim → hidden_dim]
  - Score: dot(user_emb, projected_tag_emb)
  - Loss: BPR
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

from proxytag.core.data import build_id_mappings, build_user_history, NegativeSampler, InBatchTrainDataset
from proxytag.core.utils import build_item_popularity_percentiles
from proxytag.baselines.eval_utils import evaluate_ranking


class LLMRec(pl.LightningModule):
    """
    LLM-Rec: text-only recommendation.

    Items are represented by frozen pretrained embeddings (from sentence-transformers
    applied to LLM-generated tags). Users have learned embeddings.
    """
    def __init__(
        self,
        n_users,
        n_items,
        tag_embed_dim,
        hidden_dim=64,
        dropout=0.0,
        lr=5e-4,
        weight_decay=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_users = n_users
        self.n_items = n_items
        self.hidden_dim = hidden_dim

        # User embedding (learned)
        self.user_embedding = nn.Embedding(n_users, hidden_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)

        # Tag projection (tag_dim → hidden_dim)
        self.tag_projection = nn.Sequential(
            nn.Linear(tag_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Frozen tag embeddings (set externally)
        self.item_tag_embs = None

    def set_item_tag_embs(self, item_tag_embs):
        """Set pre-computed frozen tag embeddings [n_items, tag_embed_dim]."""
        self.item_tag_embs = item_tag_embs

    def encode_user(self, user_ids):
        return self.user_embedding(user_ids)

    def encode_item(self, item_ids):
        tag_raw = self.item_tag_embs[item_ids]
        return self.tag_projection(tag_raw)

    def forward(self, user_ids, item_ids):
        user_emb = self.encode_user(user_ids)
        item_emb = self.encode_item(item_ids)
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            return self._training_step_bpr(batch)
        return self._training_step_inbatch(batch)

    def _training_step_bpr(self, batch):
        user_ids, pos_ids, neg_ids_list = batch
        batch_size = user_ids.shape[0]
        n_neg = neg_ids_list.shape[1]

        pos_scores = self(user_ids, pos_ids)
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, n_neg)
        neg_scores = self(user_ids_expanded.flatten(), neg_ids_list.flatten())
        neg_scores = neg_scores.view(batch_size, n_neg)

        diff = pos_scores.unsqueeze(1) - neg_scores
        loss = -F.logsigmoid(diff).mean()

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _training_step_inbatch(self, batch):
        user_ids, item_ids = batch
        B = user_ids.size(0)

        user_emb = self.encode_user(user_ids)
        item_emb = self.encode_item(item_ids)
        score_matrix = user_emb @ item_emb.T

        labels = torch.arange(B, device=self.device)
        loss = F.cross_entropy(score_matrix, labels)

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, pos_ids, neg_ids_list = batch
        batch_size = user_ids.shape[0]
        n_neg = neg_ids_list.shape[1]

        pos_scores = self(user_ids, pos_ids)
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, n_neg)
        neg_scores = self(user_ids_expanded.flatten(), neg_ids_list.flatten())
        neg_scores = neg_scores.view(batch_size, n_neg)

        diff = pos_scores.unsqueeze(1) - neg_scores
        loss = -F.logsigmoid(diff).mean()

        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class LLMRecDataset(Dataset):
    """Dataset for LLM-Rec training with negative sampling."""
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
        neg_items = self.neg_sampler.sample(user_id.item(), self.n_neg)
        neg_items = torch.tensor(neg_items, dtype=torch.long)
        return user_id, pos_item, neg_items


def main():
    parser = argparse.ArgumentParser(description="Train LLM-Rec baseline (text-only)")

    # Data paths
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)

    # Column names
    parser.add_argument('--user_id_col', type=str, default='userId')
    parser.add_argument('--item_id_col', type=str, default='movieId')

    # Tags (required for LLM-Rec)
    parser.add_argument('--tags_path', type=str, required=True,
                        help='Path to tag parquet (required — LLM-Rec is text-only)')
    parser.add_argument('--max_tags', type=int, default=16)

    # Model
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)

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
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--val_check_interval', type=float, default=1.0)

    # Evaluation
    parser.add_argument('--n_test_neg', type=int, default=999)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--max_eval_samples', type=int, default=None)

    # Output
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/llmrec')
    parser.add_argument('--results_dir', type=str, default='results')

    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print("=" * 80)
    print("LLM-REC BASELINE (text-only)")
    print("=" * 80)
    print(f"Job: {args.job_name}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Tags: {args.tags_path}")
    print("=" * 80)

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
    print("\n[3/5] Building user history and loading tags...")
    all_interactions = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user_history = build_user_history(all_interactions)

    # Load tag embeddings
    from proxytag.baselines.tag_utils import load_and_pool_tags
    item_tag_embs = load_and_pool_tags(
        args.tags_path, item2idx, item_id_col=args.item_id_col, max_tags=args.max_tags
    )
    tag_embed_dim = item_tag_embs.shape[1]
    print(f"  Tag embedding dim: {tag_embed_dim}")
    print(f"  Items with tags: {(item_tag_embs.abs().sum(dim=1) > 0).sum():,} / {n_items:,}")

    # Create datasets
    print("\n[4/5] Creating datasets...")
    neg_sampler_train = NegativeSampler(user_history=user_history, n_items=n_items, seed=args.seed)

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
        train_dataset = LLMRecDataset(train_df, neg_sampler_train, n_neg=args.n_train_neg)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  persistent_workers=True if args.num_workers > 0 else False)

    val_dataset = LLMRecDataset(val_df, neg_sampler_train, n_neg=args.n_val_neg)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            persistent_workers=True if args.num_workers > 0 else False)

    # Create model
    print("\n[5/5] Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LLMRec(
        n_users=n_users,
        n_items=n_items,
        tag_embed_dim=tag_embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model.set_item_tag_embs(item_tag_embs.to(device))

    # Callbacks
    os.makedirs(args.ckpt_dir, exist_ok=True)
    monitor_metric = 'val/loss'
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir, filename='best',
        monitor=monitor_metric, mode='min', save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric, patience=args.patience, mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        precision='bf16-mixed',
        gradient_clip_val=5.0,
        log_every_n_steps=50,
        val_check_interval=args.val_check_interval,
    )

    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    # Load best model for evaluation
    best_model = LLMRec.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.set_item_tag_embs(item_tag_embs.to(device))
    best_model = best_model.to(device)
    best_model.eval()

    # Evaluate
    neg_sampler_test = NegativeSampler(user_history=user_history, n_items=n_items, seed=args.seed + 1)

    def score_fn(user_ids, item_ids):
        return best_model(user_ids.to(device), item_ids.to(device))

    test_metrics, bucket_df = evaluate_ranking(
        score_fn=score_fn,
        test_df=test_df,
        neg_sampler=neg_sampler_test,
        K=args.K,
        n_neg=args.n_test_neg,
        train_df=train_df,
        n_items=n_items,
        seed=args.seed + 2,
        max_eval_samples=args.max_eval_samples,
    )

    # Print results
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
    results = {
        'job_name': args.job_name,
        'model': 'LLM-Rec',
        'tags_path': args.tags_path,
        'test_metrics': test_metrics,
        'hyperparameters': vars(args),
    }

    results_file = os.path.join(args.results_dir, f'{args.job_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    if bucket_df is not None:
        buckets_file = os.path.join(args.results_dir, f'{args.job_name}_buckets_eval.csv')
        bucket_df.to_csv(buckets_file, index=False)

    print(f"\nResults saved to: {results_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
