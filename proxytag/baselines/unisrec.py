#!/usr/bin/env python3
"""
UniSRec Baseline

Universal Sequential Recommendation with pretrained text embeddings.
Uses sentence-transformers to encode item tags, then feeds frozen embeddings
through a Mixture-of-Experts adaptor layer into a transformer encoder.

Supports transductive (item IDs + PLM embeddings) and inductive (PLM only) modes.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from proxytag.baselines.eval_utils import evaluate_ranking
from proxytag.core.data import NegativeSampler


def prepare_unisrec_data(
    train_path,
    val_path,
    test_path,
    output_dir,
    user_id_col='userId',
    item_id_col='movieId',
    timestamp_col=None,
    tags_path=None,
    tag_item_id_col=None,
    max_tags=16,
    embedding_model='all-MiniLM-L6-v2',
    tag_encoding_mode='single_string',
):
    """
    Prepare data in UniSRec's expected format:
      - {dataset}.inter: interactions
      - {dataset}.feat1CLS: pre-computed PLM embeddings (binary float32)
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = 'proxytag'

    # Load interactions
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Remap item IDs to 0-based contiguous indices (UniSRec uses token as feat file index)
    # Sort numerically so that zero-padded string tokens sort correctly in RecBole
    all_items = sorted(all_df[item_id_col].unique(), key=lambda x: str(x))
    item2idx = {item: idx for idx, item in enumerate(all_items)}
    n_items = len(all_items)
    # Zero-pad width so string sort = numeric sort (e.g., "0000", "0001", ..., "3705")
    pad_width = len(str(n_items - 1))

    # Build per-user sequences from all data (using remapped IDs)
    all_df_mapped = all_df[[user_id_col, item_id_col]].copy()
    all_df_mapped['_item_idx'] = all_df_mapped[item_id_col].map(item2idx).apply(lambda x: str(x).zfill(pad_width))
    user_sequences = all_df_mapped.groupby(user_id_col)['_item_idx'].apply(list).to_dict()

    # Create sequential format: user_id, item_id_list (history), item_id (target)
    # All item IDs are remapped to 0-based indices
    def write_sequential_inter(split_df, all_preceding_df, output_path):
        """For each interaction in split_df, the history is all items before it."""
        preceding_mapped = all_preceding_df[[user_id_col, item_id_col]].copy()
        preceding_mapped['_idx'] = preceding_mapped[item_id_col].map(item2idx).apply(lambda x: str(x).zfill(pad_width))
        user_history = preceding_mapped.groupby(user_id_col)['_idx'].apply(list).to_dict()

        split_targets = split_df[item_id_col].map(item2idx).apply(lambda x: str(x).zfill(pad_width)).values
        split_uids = split_df[user_id_col].values

        lines = ['user_id:token\titem_id_list:token_seq\titem_id:token\n']
        for i in range(len(split_uids)):
            uid = split_uids[i]
            target = split_targets[i]
            history = user_history.get(uid, [])
            if len(history) == 0:
                history = [str(0).zfill(pad_width)]
            seq_str = ' '.join(history[-50:])
            lines.append(f'{uid}\t{seq_str}\t{target}\n')
            user_history.setdefault(uid, []).append(target)

        with open(output_path, 'w') as f:
            f.writelines(lines)

    # Train: build history incrementally
    train_path_out = os.path.join(output_dir, f'{dataset_name}.train.inter')
    train_uids = train_df[user_id_col].values
    train_targets = train_df[item_id_col].map(item2idx).apply(lambda x: str(x).zfill(pad_width)).values
    user_hist_train = defaultdict(list)
    lines = ['user_id:token\titem_id_list:token_seq\titem_id:token\n']
    for i in range(len(train_uids)):
        uid = train_uids[i]
        target = train_targets[i]
        history = user_hist_train.get(uid, [])
        if len(history) == 0:
            history = [str(0).zfill(pad_width)]
        seq_str = ' '.join(history[-50:])
        lines.append(f'{uid}\t{seq_str}\t{target}\n')
        user_hist_train[uid].append(target)
    with open(train_path_out, 'w') as f:
        f.writelines(lines)

    # Valid: history = all train items for that user
    write_sequential_inter(val_df, train_df,
                           os.path.join(output_dir, f'{dataset_name}.valid.inter'))

    # Test: history = all train + val items for that user
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    write_sequential_inter(test_df, train_val_df,
                           os.path.join(output_dir, f'{dataset_name}.test.inter'))

    print(f"  Written sequential inter files (train/valid/test, {n_items} items remapped)")

    # item2idx and n_items already defined above

    # Compute PLM embeddings for items
    if tags_path:
        from proxytag.core.tags import _clean_tag_string
        from sentence_transformers import SentenceTransformer

        _tag_id_col = tag_item_id_col or item_id_col
        tags_df = pd.read_parquet(tags_path)
        item_to_tags = dict(zip(
            tags_df[_tag_id_col].astype(str),
            tags_df['tags'].fillna('')
        ))

        # Encode tags using sentence-transformers
        print(f"  Encoding tags with {embedding_model} (mode={tag_encoding_mode})...")
        st_model = SentenceTransformer(embedding_model)
        emb_dim = st_model.get_sentence_embedding_dimension()

        # Prepare texts for batch encoding
        texts = []
        text_indices = []
        for item_id, idx in item2idx.items():
            tag_str = item_to_tags.get(str(item_id), '')
            if tag_str:
                tag_str = _clean_tag_string(tag_str)
                if tag_encoding_mode == 'single_string':
                    # Encode all tags as one sentence (closer to UniSRec paper)
                    tags = [t.strip() for t in tag_str.split('|') if t.strip()][:max_tags]
                    if tags:
                        texts.append(' | '.join(tags))
                        text_indices.append(idx)
                else:
                    # Mean pool individual tag embeddings
                    tags = [t.strip() for t in tag_str.split('|') if t.strip()][:max_tags]
                    if tags:
                        texts.append(tags)
                        text_indices.append(idx)

        embeddings = np.zeros((n_items, emb_dim), dtype=np.float32)
        if tag_encoding_mode == 'single_string':
            # Batch encode all items at once
            all_vecs = st_model.encode(texts, batch_size=256, show_progress_bar=True)
            for i, idx in enumerate(text_indices):
                embeddings[idx] = all_vecs[i]
        else:
            # Mean pool per item
            for i, idx in enumerate(text_indices):
                vecs = st_model.encode(texts[i], batch_size=64, show_progress_bar=False)
                embeddings[idx] = np.mean(vecs, axis=0)

        n_with_embs = (np.abs(embeddings).sum(axis=1) > 0).sum()
        print(f"  Items with embeddings: {n_with_embs}/{n_items}")
    else:
        # No tags — use random embeddings (baseline)
        emb_dim = 384
        embeddings = np.random.randn(n_items, emb_dim).astype(np.float32) * 0.01
        print(f"  No tags provided — using random embeddings as baseline")

    # Save as binary file (UniSRec format)
    feat_path = os.path.join(output_dir, f'{dataset_name}.feat1CLS')
    embeddings.tofile(feat_path)
    print(f"  Written embeddings [{n_items}, {emb_dim}] to {feat_path}")

    return dataset_name, emb_dim, n_items


def run_unisrec(
    data_dir,
    dataset_name,
    plm_size,
    train_stage='transductive_ft',
    hidden_size=300,
    n_layers=2,
    n_heads=2,
    max_seq_length=50,
    lr=1e-3,
    epochs=100,
    batch_size=256,
    eval_K=10,
    seed=42,
):
    """Train and evaluate UniSRec."""
    # Patch torch.load for PyTorch 2.6+ compatibility
    _original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

    from recbole.config import Config
    from recbole.data import data_preparation
    from recbole.utils import init_seed, get_trainer

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from proxytag.baselines.unisrec_model import UniSRec
    from proxytag.baselines.unisrec_dataset import UniSRecDataset

    config_dict = {
        'data_path': data_dir,
        'dataset': dataset_name,

        # Model
        'model': 'UniSRec',
        'hidden_size': hidden_size,
        'inner_size': hidden_size,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'initializer_range': 0.02,
        'loss_type': 'CE',
        'MAX_ITEM_LIST_LENGTH': max_seq_length,

        # UniSRec specific
        'train_stage': train_stage,
        'temperature': 0.07,
        'lambda': 1e-3,
        'plm_suffix': 'feat1CLS',
        'plm_size': plm_size,
        'adaptor_dropout_prob': 0.2,
        'adaptor_layers': [plm_size, hidden_size],
        'n_exps': 8,

        # Training
        'learning_rate': lr,
        'epochs': epochs,
        'train_batch_size': batch_size,
        'eval_batch_size': batch_size * 4,
        'weight_decay': 0.0,
        'train_neg_sample_args': None,
        'stopping_step': 3,

        # Validation uses sampled negatives (fast)
        'eval_args': {
            'group_by': 'user',
            'order': 'TO',
            'mode': {'valid': 'uni100', 'test': 'uni100'},
        },
        'metrics': ['NDCG', 'Hit', 'MRR'],
        'topk': [eval_K],
        'valid_metric': f'NDCG@{eval_K}',
        'metric_decimal_place': 4,

        # Data
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'benchmark_filename': ['train', 'valid', 'test'],
        'alias_of_item_id': ['item_id_list'],
        'load_col': {
            'inter': ['user_id', 'item_id_list', 'item_id'],
        },

        # Logging
        'show_progress': True,
        'state': 'INFO',
    }

    config = Config(model=UniSRec, dataset=dataset_name, config_dict=config_dict)
    init_seed(seed, reproducibility=True)

    dataset = UniSRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = UniSRec(config, train_data.dataset).to(config['device'])

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True
    )

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

    torch.load = _original_torch_load

    return {
        'best_valid_score': best_valid_score,
        'valid_metrics': best_valid_result,
        'test_metrics': test_result,
        'model': model,
        'dataset': dataset,
        'config': config,
    }


def main():
    parser = argparse.ArgumentParser(description="Train UniSRec baseline")

    # Data paths
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)

    # Column names
    parser.add_argument('--user_id_col', type=str, default='userId')
    parser.add_argument('--item_id_col', type=str, default='movieId')
    parser.add_argument('--timestamp_col', type=str, default=None)

    # Tags
    parser.add_argument('--tags_path', type=str, default=None,
                        help='Path to tag parquet. Omit for random-embedding baseline.')
    parser.add_argument('--tag_item_id_col', type=str, default=None)
    parser.add_argument('--max_tags', type=int, default=16)
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--tag_encoding_mode', type=str, default='single_string',
                        choices=['single_string', 'mean_pool'],
                        help='single_string: encode all tags as one sentence (recommended). '
                             'mean_pool: encode each tag separately and average.')

    # Model
    parser.add_argument('--train_stage', type=str, default='transductive_ft',
                        choices=['inductive_ft', 'transductive_ft'],
                        help='transductive_ft: item IDs + PLM embeddings (recommended). '
                             'inductive_ft: PLM only, for cold-start/transfer scenarios.')
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--max_seq_length', type=int, default=50)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=100)

    # Evaluation
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--n_test_neg', type=int, default=999)

    # Output
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default=None)

    # System
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print("=" * 80)
    print(f"UniSRec BASELINE ({args.train_stage})")
    print("=" * 80)
    print(f"Job: {args.job_name}")
    print(f"Train stage: {args.train_stage}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Tags: {args.tags_path or 'None (random baseline)'}")
    print("=" * 80)

    # Prepare data
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = tempfile.mkdtemp(prefix='unisrec_proxytag_')

    print(f"\n[1/3] Preparing data...")
    dataset_name, plm_size, n_items = prepare_unisrec_data(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=os.path.join(data_dir, 'proxytag'),
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col,
        timestamp_col=args.timestamp_col,
        tags_path=args.tags_path,
        tag_item_id_col=args.tag_item_id_col,
        max_tags=args.max_tags,
        embedding_model=args.embedding_model,
        tag_encoding_mode=args.tag_encoding_mode,
    )

    # Train
    print(f"\n[2/3] Training UniSRec ({args.train_stage})...")
    results = run_unisrec(
        data_dir=data_dir,
        dataset_name=dataset_name,
        plm_size=plm_size,
        train_stage=args.train_stage,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_length=args.max_seq_length,
        lr=args.lr,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        eval_K=args.K,
        seed=args.seed,
    )

    # Print RecBole's own test metrics for comparison
    rb_test = results.get('test_metrics', {})
    if rb_test:
        print("\nRecBole internal test results (uni100):")
        for k, v in sorted(rb_test.items()):
            print(f"  {k}: {v}")

    # Custom evaluation
    print(f"\n[3/3] Running shared evaluation (K={args.K})...")
    dataset = results['dataset']
    model = results['model']
    recbole_config = results['config']

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    n_items_rb = dataset.item_num

    # Build mapping: original_item_id -> RecBole internal ID
    # The .inter files use remapped 0-based indices as item tokens.
    # RecBole then assigns internal IDs: token "0"->1, "1"->2, etc. (0 = padding)
    # We need: original_id -> remapped_index -> str(remapped) -> recbole_id

    # Rebuild item2idx (original -> 0-based remapped) from the raw data
    all_items_raw = pd.concat([
        pd.read_parquet(args.train_path),
        pd.read_parquet(args.val_path),
        pd.read_parquet(args.test_path),
    ], ignore_index=True)[args.item_id_col].unique()
    all_items_sorted = sorted(all_items_raw, key=lambda x: str(x))
    item2idx = {item: idx for idx, item in enumerate(all_items_sorted)}
    pad_width = len(str(len(all_items_sorted) - 1))

    # RecBole's token-to-internal-id mapping (remapped token string -> recbole_id)
    item_token2id_raw = dataset.field2token_id[iid_field]
    if isinstance(item_token2id_raw, dict):
        token2rbid = item_token2id_raw
    else:
        token2rbid = {str(token): idx for idx, token in enumerate(item_token2id_raw)}
        token2rbid.pop('[PAD]', None)

    # Composite mapping: original_item_id -> recbole_internal_id
    def orig_item_to_rbid(orig_id):
        remapped = item2idx.get(orig_id)
        if remapped is None:
            return -1
        return token2rbid.get(str(remapped).zfill(pad_width), -1)

    # User mapping: user tokens in .inter are original user IDs
    user_token2id_raw = dataset.field2token_id[uid_field]
    if isinstance(user_token2id_raw, dict):
        user_token2id = user_token2id_raw
    else:
        user_token2id = {str(token): idx for idx, token in enumerate(user_token2id_raw)}
        user_token2id.pop('[PAD]', None)

    test_raw = pd.read_parquet(args.test_path)
    test_df_eval = pd.DataFrame({
        'uid': test_raw[args.user_id_col].map(lambda x: user_token2id.get(str(x), -1)),
        'iid': test_raw[args.item_id_col].map(lambda x: orig_item_to_rbid(x)),
    })
    test_df_eval = test_df_eval[(test_df_eval['uid'] > 0) & (test_df_eval['iid'] > 0)]

    all_raw = pd.concat([
        pd.read_parquet(args.train_path),
        pd.read_parquet(args.val_path),
        pd.read_parquet(args.test_path),
    ], ignore_index=True)
    all_raw['_uid'] = all_raw[args.user_id_col].astype(str).map(user_token2id).fillna(-1).astype(int)
    all_raw['_iid'] = all_raw[args.item_id_col].map(lambda x: orig_item_to_rbid(x))
    valid_raw = all_raw[(all_raw['_uid'] > 0) & (all_raw['_iid'] > 0)]
    user_history = valid_raw.groupby('_uid')['_iid'].apply(set).to_dict()

    class RecBoleNegSampler:
        def __init__(self, user_history, n_items, seed=42):
            self.user_history = user_history
            self.n_items = n_items
            self.rng = np.random.default_rng(seed)

        def sample(self, user_id, n_neg):
            interacted = self.user_history.get(user_id, set())
            negs = []
            seen = set()
            max_attempts = n_neg * 10
            attempts = 0
            while len(negs) < n_neg and attempts < max_attempts:
                candidates = self.rng.integers(1, self.n_items, size=n_neg * 2)
                for c in candidates:
                    if c not in interacted and c not in seen:
                        negs.append(int(c))
                        seen.add(c)
                        if len(negs) == n_neg:
                            break
                attempts += 1
            return negs

    neg_sampler = RecBoleNegSampler(user_history=user_history, n_items=n_items_rb, seed=args.seed + 1)

    # Build item popularity buckets from training data (using RecBole internal IDs)
    from proxytag.core.utils import build_item_popularity_percentiles
    train_raw = pd.read_parquet(args.train_path)
    train_rb = pd.DataFrame({
        'iid': train_raw[args.item_id_col].map(lambda x: orig_item_to_rbid(x))
    })
    train_rb = train_rb[train_rb['iid'] > 0]
    _, item_bucket = build_item_popularity_percentiles(train_rb, n_items_rb, n_bins=10)

    # Sequential evaluation with growing histories:
    # For each test interaction (in order), the user's history includes
    # train+val items plus all preceding test items — matching what other
    # baselines see and how the .inter files were constructed.
    from recbole.data.interaction import Interaction
    max_seq_len = recbole_config['MAX_ITEM_LIST_LENGTH']

    # Build initial user sequences from train+val
    train_val_raw = pd.concat([pd.read_parquet(args.train_path), pd.read_parquet(args.val_path)], ignore_index=True)
    train_val_raw['_uid'] = train_val_raw[args.user_id_col].astype(str).map(user_token2id).fillna(-1).astype(int)
    train_val_raw['_iid'] = train_val_raw[args.item_id_col].map(lambda x: orig_item_to_rbid(x))
    valid_tv = train_val_raw[(train_val_raw['_uid'] > 0) & (train_val_raw['_iid'] > 0)]
    user_seq = {uid: list(items) for uid, items in valid_tv.groupby('_uid')['_iid'].apply(list).items()}

    eval_device = next(model.parameters()).device
    model.eval()

    def _score_all_items(seq_list):
        """Score all items for a single user given their history."""
        seq = seq_list[-max_seq_len:]
        seq_len = len(seq)
        if seq_len == 0:
            seq = [0]
            seq_len = 0
        padded = seq + [0] * (max_seq_len - len(seq))
        with torch.no_grad():
            interaction = Interaction({
                'item_id_list': torch.tensor([padded], dtype=torch.long),
                'item_length': torch.tensor([max(seq_len, 1)], dtype=torch.long),
            }).to(eval_device)
            return model.full_sort_predict(interaction).squeeze(0).cpu()

    # Run evaluation with growing histories
    K = args.K
    n_neg = args.n_test_neg
    rng = np.random.default_rng(args.seed + 2)
    tie_rng = np.random.default_rng(args.seed + 2)

    rows = test_df_eval[['uid', 'iid']].values
    ndcgs, recalls, precisions = [], [], []
    from collections import defaultdict
    by_bucket = defaultdict(lambda: {"precision": [], "recall": [], "ndcg": [], "n": 0})
    n_candidates = n_neg + 1

    from tqdm import tqdm
    for idx in tqdm(range(len(rows)), desc="Evaluating", file=sys.stdout):
        u, pos_i = int(rows[idx, 0]), int(rows[idx, 1])

        # Score all items for this user's current history
        seq_list = user_seq.get(u, [])
        all_scores = _score_all_items(seq_list).numpy()

        # Sample negatives
        neg_items = neg_sampler.sample(u, n_neg)

        # Gather scores for positive + negatives
        candidate_ids = [pos_i] + neg_items
        candidate_scores = all_scores[candidate_ids]
        candidate_scores += tie_rng.random(n_candidates) * 1e-8

        # Rank of positive (index 0)
        pos_score = candidate_scores[0]
        rank = (candidate_scores > pos_score).sum()
        hit = 1.0 if rank < K else 0.0
        ndcg_val = 1.0 / np.log2(rank + 2) if hit else 0.0
        recall_val = hit
        prec_val = hit / K
        ndcgs.append(ndcg_val)
        recalls.append(recall_val)
        precisions.append(prec_val)

        b = int(item_bucket[pos_i])
        by_bucket[b]["ndcg"].append(ndcg_val)
        by_bucket[b]["recall"].append(recall_val)
        by_bucket[b]["precision"].append(prec_val)
        by_bucket[b]["n"] += 1

        # Grow history: append this positive item for subsequent interactions
        user_seq.setdefault(u, []).append(pos_i)

    test_metrics = {
        f'ndcg@{K}': float(np.mean(ndcgs)),
        f'recall@{K}': float(np.mean(recalls)),
        f'precision@{K}': float(np.mean(precisions)),
        f'ndcg@{K}_std': float(np.std(ndcgs)),
        f'recall@{K}_std': float(np.std(recalls)),
        f'precision@{K}_std': float(np.std(precisions)),
        'n_eval': len(rows),
    }

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

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    for metric, value in sorted(test_metrics.items()):
        if not metric.endswith('_std') and metric != 'n_eval':
            print(f"  {metric}: {value:.4f}")
    print(f"  N eval: {test_metrics['n_eval']:,}")

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    results_out = {
        'job_name': args.job_name,
        'model': f'UniSRec_{args.train_stage}',
        'tags_path': args.tags_path,
        'test_metrics': test_metrics,
        'hyperparameters': vars(args),
    }

    results_file = os.path.join(args.results_dir, f'{args.job_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_out, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    if bucket_df is not None and len(bucket_df) > 0:
        bucket_file = os.path.join(args.results_dir, f'{args.job_name}_buckets_eval.csv')
        bucket_df.to_csv(bucket_file, index=False)
        print(f"Bucket eval saved to: {bucket_file}")

    if not args.data_dir and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    print("=" * 80)


if __name__ == '__main__':
    main()
