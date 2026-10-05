#!/usr/bin/env python3
"""
KAR (Knowledge-Augmented Recommendation) Baseline with FiBiNet

Based on: "Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models"
- Uses FiBiNet as the backbone CTR model
- Integrates LLM-generated tag embeddings via ConvertNet adapter
- Tags are converted from semantic space to CF space via learned MLP
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from proxytag.core.data import build_id_mappings, build_user_history, NegativeSampler, InBatchTrainDataset
from proxytag.core.tags import build_item_tag_tensors
from proxytag.baselines.eval_utils import evaluate_ranking


class SqueezeExtractionLayer(nn.Module):
    """SENET layer from FiBiNet."""
    def __init__(self, num_fields, reduction_ratio):
        super().__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_size, num_fields, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


class BilinearInteractionLayer(nn.Module):
    """Bilinear interaction layer from FiBiNet."""
    def __init__(self, embed_size, num_fields):
        super().__init__()
        self.bilinear_layer = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        bilinear_list = []
        for i in range(len(feature_emb_list)):
            for j in range(i + 1, len(feature_emb_list)):
                v_i = feature_emb_list[i]
                v_j = feature_emb_list[j]
                bilinear_list.append(self.bilinear_layer(v_i) * v_j)
        return torch.cat(bilinear_list, dim=1)


class ConvertNet(nn.Module):
    """Adapter network: converts tag embeddings from semantic space to CF space."""
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FiBiNetKAR(pl.LightningModule):
    """
    FiBiNet with Knowledge Augmentation (KAR).

    Architecture:
    1. User/Item ID embeddings (CF space)
    2. Tag embeddings converted via ConvertNet (semantic → CF space)
    3. FiBiNet bilinear interactions on combined features
    4. MLP prediction head
    """
    def __init__(
        self,
        n_users,
        n_items,
        tag_emb_tensors,
        tag_mask_tensor,
        embed_dim=32,
        reduction_ratio=0.5,
        convert_hidden_dims=[128, 32],
        convert_dropout=0.0,
        final_mlp_dims=[200, 80],
        dropout=0.0,
        lr=5e-4,
        weight_decay=0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tag_emb_tensors', 'tag_mask_tensor'])

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.register_buffer('tag_emb_tensors', tag_emb_tensors)
        self.register_buffer('tag_mask_tensor', tag_mask_tensor)

        # ID embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # Tag conversion: semantic → CF space
        tag_emb_dim = tag_emb_tensors.shape[2]
        self.tag_pooling = nn.Linear(tag_emb_dim, embed_dim)
        self.convert_net = ConvertNet(embed_dim, convert_hidden_dims, convert_dropout)

        # FiBiNet: 3 fields (user, item, tags)
        self.num_fields = 3
        self.senet_layer = SqueezeExtractionLayer(self.num_fields, reduction_ratio)
        self.bilinear_layer = BilinearInteractionLayer(embed_dim, self.num_fields)

        # Final MLP
        bilinear_dim = self.num_fields * (self.num_fields - 1) * embed_dim
        mlp_layers = []
        prev_dim = bilinear_dim
        for hidden_dim in final_mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.final_mlp = nn.Sequential(*mlp_layers)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def get_tag_representation(self, item_ids):
        tag_embs = self.tag_emb_tensors[item_ids]
        tag_masks = self.tag_mask_tensor[item_ids]
        tag_masks_expanded = tag_masks.unsqueeze(-1)
        masked_embs = tag_embs * tag_masks_expanded
        pooled = masked_embs.sum(dim=1) / (tag_masks.sum(dim=1, keepdim=True) + 1e-8)
        tag_repr = self.tag_pooling(pooled)
        tag_repr = self.convert_net(tag_repr)
        return tag_repr

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        tag_repr = self.get_tag_representation(item_ids)

        if tag_repr.shape[1] != self.embed_dim:
            tag_repr = F.pad(tag_repr, (0, self.embed_dim - tag_repr.shape[1]))

        feat_embed = torch.stack([user_emb, item_emb, tag_repr], dim=1)

        senet_embed = self.senet_layer(feat_embed)
        bilinear_p = self.bilinear_layer(feat_embed)
        bilinear_q = self.bilinear_layer(senet_embed)

        combined = torch.cat([bilinear_p, bilinear_q], dim=1)
        combined_flat = combined.flatten(start_dim=1)
        logits = self.final_mlp(combined_flat).squeeze(-1)
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
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


class KARDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="Train KAR baseline (FiBiNet + ConvertNet)")

    # Data paths
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)

    # Column names
    parser.add_argument('--user_id_col', type=str, default='userId')
    parser.add_argument('--item_id_col', type=str, default='movieId')

    # Tags (required for KAR)
    parser.add_argument('--tags_path', type=str, required=True,
                        help='Path to tag parquet (required — KAR needs tags)')
    parser.add_argument('--max_tags', type=int, default=16)

    # Model
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--reduction_ratio', type=float, default=0.5)
    parser.add_argument('--convert_hidden_dims', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--convert_dropout', type=float, default=0.0)
    parser.add_argument('--final_mlp_dims', type=int, nargs='+', default=[200, 80])
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
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--val_check_interval', type=float, default=1.0)

    # Evaluation
    parser.add_argument('--n_test_neg', type=int, default=999)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--max_eval_samples', type=int, default=None)

    # Output
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/kar')
    parser.add_argument('--results_dir', type=str, default='results')

    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print("=" * 80)
    print("KAR BASELINE (FiBiNet + ConvertNet)")
    print("=" * 80)
    print(f"Job: {args.job_name}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"ConvertNet: {args.convert_hidden_dims}")
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
    tag_emb_tensors, tag_mask_tensor = build_item_tag_tensors(
        items_df_path=args.tags_path,
        item2idx=item2idx,
        item_id_col=args.item_id_col,
        max_tags=args.max_tags,
        model_name='all-MiniLM-L6-v2',
        overwrite=False,
    )
    print(f"  Tag embedding shape: {tag_emb_tensors.shape}")

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
        train_dataset = KARDataset(train_df, neg_sampler_train, n_neg=args.n_train_neg)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  persistent_workers=True if args.num_workers > 0 else False)

    val_dataset = KARDataset(val_df, neg_sampler_train, n_neg=args.n_val_neg)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            persistent_workers=True if args.num_workers > 0 else False)

    # Create model
    print("\n[5/5] Training model...")
    model = FiBiNetKAR(
        n_users=n_users,
        n_items=n_items,
        tag_emb_tensors=tag_emb_tensors,
        tag_mask_tensor=tag_mask_tensor,
        embed_dim=args.embed_dim,
        reduction_ratio=args.reduction_ratio,
        convert_hidden_dims=args.convert_hidden_dims,
        convert_dropout=args.convert_dropout,
        final_mlp_dims=args.final_mlp_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
    best_model = FiBiNetKAR.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        tag_emb_tensors=tag_emb_tensors,
        tag_mask_tensor=tag_mask_tensor,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = best_model.to(device)
    best_model.eval()

    # Move tag tensors to CPU during eval to avoid GPU OOM on large datasets
    best_model.tag_emb_tensors = best_model.tag_emb_tensors.cpu()
    best_model.tag_mask_tensor = best_model.tag_mask_tensor.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Evaluate
    neg_sampler_test = NegativeSampler(user_history=user_history, n_items=n_items, seed=args.seed + 1)

    def score_fn(user_ids, item_ids):
        chunk_size = 2048
        scores = []
        for i in range(0, len(user_ids), chunk_size):
            u_chunk = user_ids[i:i+chunk_size].to(device)
            i_chunk = item_ids[i:i+chunk_size].to(device)
            # Fetch tag data for this chunk only, move to GPU on the fly
            tag_embs = best_model.tag_emb_tensors[i_chunk.cpu()].to(device)
            tag_masks = best_model.tag_mask_tensor[i_chunk.cpu()].to(device)
            tag_masks_expanded = tag_masks.unsqueeze(-1)
            masked_embs = tag_embs * tag_masks_expanded
            pooled = masked_embs.sum(dim=1) / (tag_masks.sum(dim=1, keepdim=True) + 1e-8)
            tag_repr = best_model.tag_pooling(pooled)
            tag_repr = best_model.convert_net(tag_repr)
            if tag_repr.shape[1] != best_model.embed_dim:
                tag_repr = F.pad(tag_repr, (0, best_model.embed_dim - tag_repr.shape[1]))

            user_emb = best_model.user_embedding(u_chunk)
            item_emb = best_model.item_embedding(i_chunk)
            feat_embed = torch.stack([user_emb, item_emb, tag_repr], dim=1)
            senet_embed = best_model.senet_layer(feat_embed)
            bilinear_p = best_model.bilinear_layer(feat_embed)
            bilinear_q = best_model.bilinear_layer(senet_embed)
            combined = torch.cat([bilinear_p, bilinear_q], dim=1)
            combined_flat = combined.flatten(start_dim=1)
            s = best_model.final_mlp(combined_flat).squeeze(-1)
            scores.append(s.cpu())
        return torch.cat(scores).to(user_ids.device)

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

    epochs_trained = early_stop_callback.stopped_epoch + 1 if early_stop_callback.stopped_epoch > 0 else trainer.current_epoch + 1
    stopped_early = early_stop_callback.stopped_epoch > 0
    best_epoch = epochs_trained - args.patience if stopped_early else epochs_trained
    best_val_loss = float(checkpoint_callback.best_model_score)
    final_train_loss = float(trainer.callback_metrics.get('train/loss_epoch', 0))
    final_val_loss = float(trainer.callback_metrics.get('val/loss', 0))
    train_val_gap = final_val_loss - final_train_loss

    results = {
        'job_name': args.job_name,
        'model': 'KAR',
        'tags_path': args.tags_path,
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

    print(f"\nResults saved to: {results_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
