import argparse
import os
import gc
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from proxytag.core.data import load_df, build_id_mappings, build_user_history, NegativeSampler, RecDataModule
from proxytag.core.tags import build_item_tag_tensors
from proxytag.core.model import HybridRecLightning
from proxytag.core.eval import eval_split_by_bucket
from proxytag.core.utils import build_item_popularity_percentiles


def parse_args():
    p = argparse.ArgumentParser()

    # job
    p.add_argument("--job_name", type=str, required=True)
    
    # paths
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--items_data_path", type=str, required=False,
                   help='Required if use_tags=True, optional for CF-only')

    # column names
    p.add_argument("--user_id_col", type=str, default="userId")
    p.add_argument("--item_id_col", type=str, default="movieId")

    # embeddings
    p.add_argument("--sentence_transformers_model", type=str, required=True)
    p.add_argument("--max_tags", type=int, default=24)

    # negative sampling
    p.add_argument("--negative_sampler_n_items_train", type=int, default=4)
    p.add_argument("--negative_sampler_n_items_test", type=int, default=999)

    # model
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)

    # cross-att not implemented in this stable baseline
    p.add_argument("--max_history_items", type=int, default=0)  # kept for CLI compatibility

    # training
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--embedding_reg", type=float, default=0.0)

    # evaluation
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--val_check_interval", type=float, default=1.0,
                   help="Run validation every N fraction of epoch (e.g., 0.1 = 10x per epoch)")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Max interactions for val/test evaluation (uniformly sampled)")

    # infra
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--results_dir", type=str, default="results")

    # ablations
    p.add_argument("--use_item_id", action="store_true", default=True)
    p.add_argument("--no_use_item_id", action="store_false", dest="use_item_id")
    p.add_argument("--use_tags", action="store_true", default=True)
    p.add_argument("--no_use_tags", action="store_false", dest="use_tags")

    # could be either cross_encoder or mean_pooling
    p.add_argument("--tag_encoding_type", type=str, default="cross_encoder")  

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # Validate arguments
    if args.use_tags and not args.items_data_path:
        raise ValueError("--items_data_path is required when use_tags=True")

    print("use_tags is ", args.use_tags)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Better performance on Tensor Core GPUs (A10G)
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    # Load splits
    train_df = load_df(args.train_path)
    val_df = load_df(args.val_path)
    test_df = load_df(args.test_path)

    # Map IDs across all splits
    train_df, val_df, test_df, user2idx, idx2user, item2idx, idx2item = build_id_mappings(
        train_df, val_df, test_df,
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col
    )
    num_users = len(user2idx)
    num_items = len(item2idx)

    # Build user history across all splits (for valid negative sampling)
    user_history = build_user_history(train_df, val_df, test_df)

    # Build/load tag tensors (only if using tags)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.use_tags:
        item_tag_embs, item_tag_mask = build_item_tag_tensors(
            items_df_path=args.items_data_path,
            item2idx=item2idx,
            model_name=args.sentence_transformers_model,
            max_tags=args.max_tags,
            device="cpu",  # recommended: keep as CPU buffers; Lightning will move model separately
            normalize=True,
            overwrite=False,
            item_id_col=args.item_id_col,
        )
    else:
        # CF-only: create dummy tensors (not used by model)
        item_tag_embs = torch.zeros((num_items, 1, 384))
        item_tag_mask = torch.zeros((num_items, 1), dtype=torch.bool)

    # Negative samplers
    neg_sampler_train = NegativeSampler(num_items, user_history, seed=args.seed)
    neg_sampler_eval = NegativeSampler(num_items, user_history, seed=args.seed + 1)

    # DataModule
    dm = RecDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        neg_sampler_train=neg_sampler_train,
        batch_size=args.batch_size,
        n_neg_train=args.negative_sampler_n_items_train,
        num_workers=args.num_workers,
    )
    # attach eval sampler for model hooks
    dm.neg_sampler_eval = neg_sampler_eval

    # Model
    model = HybridRecLightning(
        num_users=num_users,
        num_items=num_items,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        lr=args.lr,
        use_item_id=args.use_item_id,
        use_tags=args.use_tags,
        n_neg_eval=args.negative_sampler_n_items_test,
        K=args.K,
        seed=args.seed,
        embedding_reg=args.embedding_reg,
        tag_encoding_type=args.tag_encoding_type,
        max_eval_samples=args.max_eval_samples,
    )
    model.set_item_tag_tensors(item_tag_embs, item_tag_mask)

    monitor_key = f"val/ndcg@{args.K}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        monitor=monitor_key,
        mode="max",
        save_top_k=1,
        filename="best",
        verbose=True,
    )

    early_stop_cb = EarlyStopping(
        monitor=monitor_key,
        mode="max",
        patience=3,
        min_delta=1e-4,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        gradient_clip_val=5.0,
        log_every_n_steps=50,
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=0,  # important: avoids dummy val sanity issues
    )

    trainer.fit(model, datamodule=dm)  # , ckpt_path=ckpt_path)

    # Final test evaluation (manual)
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    best_model = HybridRecLightning.load_from_checkpoint(
        checkpoint_cb.best_model_path
    )
    best_model.set_item_tag_tensors(item_tag_embs, item_tag_mask)

    best_model.eval()
    best_model = best_model.to(device)

    item_count, item_bucket = build_item_popularity_percentiles(train_df, num_items, n_bins=10)

    # Create fresh negative sampler for test evaluation (seed+1 matches popularity baseline)
    # The original neg_sampler_eval has been consumed by validation epochs,
    # so its RNG state has diverged — use a fresh one for reproducible comparison.
    neg_sampler_test = NegativeSampler(num_items, user_history, seed=args.seed + 1)

    test_metrics, test_bucket_df = eval_split_by_bucket(
        model=best_model,
        test_df=test_df,
        neg_sampler=neg_sampler_test,
        item_bucket=item_bucket,
        n_items=num_items,
        n_neg=args.negative_sampler_n_items_test,
        K=args.K,
        seed=args.seed + 2,
        max_eval_samples=args.max_eval_samples,
        debug=True,  # Enable debug output
    )

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    buckets_path = os.path.join(args.results_dir, f'{args.job_name}_buckets_eval.csv')

    test_bucket_df.to_csv(buckets_path, index=False)

    print(f"\n{'='*80}")
    print(f"FINAL TEST RESULTS - {args.job_name}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Items data path: {args.items_data_path}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  N heads: {args.n_heads}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Embedding reg: {args.embedding_reg}")
    print(f"  Use item ID: {args.use_item_id}")
    print(f"  Use tags: {args.use_tags}")
    print(f"  Train negatives: {args.negative_sampler_n_items_train}")
    print(f"  Test negatives: {args.negative_sampler_n_items_test}")
    print(f"  K: {args.K}")
    print(f"\nResults:")
    print(f"  Precision@{args.K}: {test_metrics['precision@K']:.4f}")
    print(f"  Recall@{args.K}: {test_metrics['recall@K']:.4f}")
    print(f"  NDCG@{args.K}: {test_metrics['ndcg@K']:.4f}")
    print(f"  N eval: {test_metrics['n_eval']}")
    print(f"{'='*80}\n")

    # Save results to JSON
    results_dict = {
        'job_name': args.job_name,
        'items_data_path': args.items_data_path,
        'hidden_dim': args.hidden_dim,
        'n_heads': args.n_heads,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_epochs': args.max_epochs,
        'embedding_reg': args.embedding_reg,
        'use_item_id': args.use_item_id,
        'use_tags': args.use_tags,
        'train_negatives': args.negative_sampler_n_items_train,
        'test_negatives': args.negative_sampler_n_items_test,
        'K': args.K,
        f'precision@{args.K}': test_metrics['precision@K'],
        f'recall@{args.K}': test_metrics['recall@K'],
        f'ndcg@{args.K}': test_metrics['ndcg@K'],
        'n_eval': test_metrics['n_eval'],
    }

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, f'{args.job_name}_results.json')
    buckets_path = os.path.join(args.results_dir, f'{args.job_name}_buckets_eval.csv')

    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved to {results_path}")
    print(f"Bucket results saved to {buckets_path}")


if __name__ == "__main__":
    main()
