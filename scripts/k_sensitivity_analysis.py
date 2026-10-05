#!/usr/bin/env python3
"""
K sensitivity analysis for NO@K proxy metric.

For each dataset (amazon-books, amazon-movies), computes NO@K at K=5,10,20,50
for all tag variants, then correlates with downstream NDCG@10 per model.
Produces a figure and a CSV with raw numbers.

Optimized: loads SentenceTransformer once, computes all K values per item in one pass.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
if not hasattr(np.__config__, 'get_info'):
    np.__config__.get_info = lambda x: {}
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from proxytag.core.data import load_df, build_id_mappings
from proxytag.analysis.proxy_metric import (
    sample_panel_items, get_cf_item_emb,
)

# ============================================================================
# Configuration
# ============================================================================

DATASETS = {
    "amazon-books": {
        "train_path": "data/amazon-books/interactions/train.parquet",
        "user_id_col": "userId",
        "item_id_col": "asin",
        "tags_dir": "data/tags/amazon-books/full",
        "downstream_csv": "results/amazon-books_nok_vs_downstream.csv",
    },
    "amazon-movies": {
        "train_path": "data/amazon-movies/interactions/train.parquet",
        "user_id_col": "userId",
        "item_id_col": "asin",
        "tags_dir": "data/tags/amazon-movies/full",
        "downstream_csv": "results/amazon-movies_nok_vs_downstream.csv",
    },
}

K_VALUES = [5, 10, 20, 50]
PANEL_SIZE = 0.05
CF_METHOD = 'bpr'
SENTENCE_MODEL = 'all-MiniLM-L6-v2'
MAX_TAGS = 16

# Model display names and which CSV column prefix they use
MODEL_DISPLAY = {
    'AutoInt': 'AutoInt',
    'DCNv2': 'DCNv2',
    'KAR': 'KAR',
    'LLM-Rec': 'LLM-Rec',
    'UniSRec': 'UniSRec_transductive_ft',
}

RESULTS_DIR = 'results'
FIGURES_DIR = 'paper/figures'


def compute_nok_multi_k_single_item(panel_item_idx, panel_cf_vectors, panel_tag_embeddings, k_values):
    """
    Compute NO@K for multiple K values in a single pass for one panel item.

    Returns dict: {K: recall}.
    """
    cf_vec = panel_cf_vectors[panel_item_idx]
    tag_vec = panel_tag_embeddings[panel_item_idx]

    cf_sims = panel_cf_vectors @ cf_vec
    tag_sims = panel_tag_embeddings @ tag_vec

    cf_sims[panel_item_idx] = -np.inf
    tag_sims[panel_item_idx] = -np.inf

    max_k = max(k_values)
    max_k = min(max_k, len(panel_cf_vectors) - 1)

    cf_sorted = np.argsort(-cf_sims)[:max_k]
    tag_sorted = np.argsort(-tag_sims)[:max_k]

    results = {}
    for K in k_values:
        K_eff = min(K, len(panel_cf_vectors) - 1)
        cf_top_k = set(cf_sorted[:K_eff].tolist())
        tag_top_k = set(tag_sorted[:K_eff].tolist())
        results[K] = len(cf_top_k & tag_top_k) / K_eff

    return results


def encode_tags_batch(items_df, item_col, tag_col, st_model, max_tags=16):
    """
    Encode tag embeddings using a pre-loaded SentenceTransformer model.
    Collects all unique tags, encodes them in one batch, then computes per-item means.

    Returns:
        item_tag_vectors: [num_items, emb_dim] averaged tag embeddings (L2-normalized)
        item_to_idx: Mapping from item ID to index
    """
    item_to_idx = {iid: i for i, iid in enumerate(items_df[item_col].values)}
    n_items = len(items_df)
    emb_dim = st_model.get_sentence_embedding_dimension()

    # Collect all tags per item
    item_tags = []
    all_unique_tags = set()
    for _, row in items_df.iterrows():
        tags_str = row[tag_col]
        if pd.isna(tags_str) or not tags_str:
            item_tags.append([])
            continue
        if '|' in tags_str:
            tags = [t.strip() for t in tags_str.split('|') if t.strip()][:max_tags]
        else:
            tags = [t.strip() for t in tags_str.split(',') if t.strip()][:max_tags]
        item_tags.append(tags)
        all_unique_tags.update(tags)

    unique_tags = sorted(all_unique_tags)
    if not unique_tags:
        return np.zeros((n_items, emb_dim), dtype=np.float32), item_to_idx

    # Encode all unique tags in one batch
    print(f"    Encoding {len(unique_tags)} unique tags in batch...")
    tag_to_emb = {}
    batch_size = 512
    for start in range(0, len(unique_tags), batch_size):
        batch = unique_tags[start:start + batch_size]
        embs = st_model.encode(batch, convert_to_numpy=True, show_progress_bar=False,
                               normalize_embeddings=True)
        for tag, emb in zip(batch, embs):
            tag_to_emb[tag] = emb

    # Build per-item averaged embeddings
    item_tag_vectors = np.zeros((n_items, emb_dim), dtype=np.float32)
    for i, tags in enumerate(item_tags):
        if not tags:
            continue
        tag_embs = [tag_to_emb[t] for t in tags if t in tag_to_emb]
        if tag_embs:
            item_tag_vectors[i] = np.mean(tag_embs, axis=0)

    # L2-normalize
    norms = np.linalg.norm(item_tag_vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    item_tag_vectors = item_tag_vectors / norms

    n_with_tags = (np.linalg.norm(item_tag_vectors, axis=1) > 0).sum()
    print(f"    Items with tags: {n_with_tags:,} / {n_items:,}")

    return item_tag_vectors, item_to_idx


def find_ndcg_col(df, model_prefix):
    """Find the ndcg@10 column for a given model prefix."""
    for col in df.columns:
        if col.startswith(model_prefix) and 'ndcg@10' in col:
            return col
    return None


def compute_nok_multi_k(dataset_name, k_values):
    """
    Compute NO@K at multiple K values for all tag variants in a dataset.

    Returns a dict: {variant: {K: mean_nok}}.
    """
    cfg = DATASETS[dataset_name]
    tag_files = sorted(glob.glob(os.path.join(cfg["tags_dir"], "*.parquet")))

    if not tag_files:
        print(f"  No tag files found in {cfg['tags_dir']}")
        return {}

    # Load training data
    print(f"  Loading training data from {cfg['train_path']}...")
    train_df = load_df(cfg["train_path"])
    print(f"  Training interactions: {len(train_df):,}")

    # Build ID mappings
    val_df = train_df.head(100)
    test_df = train_df.head(100)
    _, _, _, user2idx, idx2user, item2idx, idx2item = build_id_mappings(
        train_df, val_df, test_df,
        user_id_col=cfg["user_id_col"],
        item_id_col=cfg["item_id_col"]
    )
    num_items = len(item2idx)

    # Sample panel items (deterministic)
    panel_items, item_bins = sample_panel_items(
        train_df, cfg["item_id_col"], panel_size=PANEL_SIZE, n_bins=10, seed=42
    )

    # Train BPR model (once)
    print(f"  Training CF model ({CF_METHOD.upper()})...")
    cf_emb_dict = get_cf_item_emb(
        train_df, item2idx, cfg["user_id_col"], cfg["item_id_col"],
        cf_method=CF_METHOD,
    )
    emb_dim = next(iter(cf_emb_dict.values())).shape[0]
    cf_embeddings = np.zeros((num_items, emb_dim), dtype=np.float32)
    for item_id, emb in cf_emb_dict.items():
        if item_id in item2idx:
            cf_embeddings[item2idx[item_id]] = emb

    # Load SentenceTransformer model ONCE
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Loading SentenceTransformer model ({SENTENCE_MODEL}) on {device}...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(SENTENCE_MODEL, device=device)

    # For each tag variant, compute NO@K at all K values
    results = {}  # {variant: {K: mean_nok}}

    for tag_file in tqdm(tag_files, desc="Processing variants"):
        variant = os.path.splitext(os.path.basename(tag_file))[0]

        items_df = pd.read_parquet(tag_file)

        tag_embeddings, tag_item2idx = encode_tags_batch(
            items_df, cfg["item_id_col"], 'tags',
            st_model, max_tags=MAX_TAGS,
        )

        # Align tag embeddings to CF item ordering
        tag_dim = tag_embeddings.shape[1]
        aligned_tag_embeddings = np.zeros((num_items, tag_dim), dtype=np.float32)
        for item_id, tag_idx in tag_item2idx.items():
            if item_id in item2idx:
                aligned_tag_embeddings[item2idx[item_id]] = tag_embeddings[tag_idx]

        # Build panel index arrays (filter to items with tags)
        panel_indices = []
        for item_id in panel_items:
            if item_id in item2idx:
                idx = item2idx[item_id]
                if np.linalg.norm(aligned_tag_embeddings[idx]) > 1e-6:
                    panel_indices.append(idx)
        panel_indices = np.array(panel_indices)

        # Extract panel embeddings
        panel_cf_emb = cf_embeddings[panel_indices]
        panel_tag_emb = aligned_tag_embeddings[panel_indices]

        # Compute NO@K at each K value for all panel items in one pass
        k_to_recalls = {K: [] for K in k_values}
        for i in range(len(panel_indices)):
            item_results = compute_nok_multi_k_single_item(
                i, panel_cf_emb, panel_tag_emb, k_values
            )
            for K in k_values:
                k_to_recalls[K].append(item_results[K])

        variant_results = {K: float(np.mean(k_to_recalls[K])) for K in k_values}
        results[variant] = variant_results

    return results


def run_analysis():
    """Run K sensitivity analysis for both datasets and generate figure + CSV."""
    all_rows = []

    for dataset_name, cfg in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"K SENSITIVITY ANALYSIS: {dataset_name}")
        print(f"{'='*80}")

        # Check for cached NOK results to avoid recomputation
        cache_path = os.path.join(RESULTS_DIR, f'{dataset_name}_k_sensitivity_nok_cache.json')

        if os.path.exists(cache_path):
            print(f"  Loading cached NO@K from {cache_path}")
            with open(cache_path) as f:
                nok_multi_k = json.load(f)
            # Convert string keys back to int
            nok_multi_k = {
                variant: {int(k): v for k, v in kv.items()}
                for variant, kv in nok_multi_k.items()
            }
        else:
            nok_multi_k = compute_nok_multi_k(dataset_name, K_VALUES)
            # Cache results
            os.makedirs(RESULTS_DIR, exist_ok=True)
            cache_data = {
                variant: {str(k): v for k, v in kv.items()}
                for variant, kv in nok_multi_k.items()
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"  Cached NO@K results to {cache_path}")

        if not nok_multi_k:
            print(f"  No results for {dataset_name}, skipping.")
            continue

        # Load downstream NDCG@10 from CSV
        downstream_csv = cfg["downstream_csv"]
        if not os.path.exists(downstream_csv):
            print(f"  Downstream CSV not found: {downstream_csv}")
            continue

        downstream_df = pd.read_csv(downstream_csv)
        print(f"  Loaded downstream metrics: {len(downstream_df)} variants")

        # For each model and each K, compute Spearman correlation
        for display_name, col_prefix in MODEL_DISPLAY.items():
            ndcg_col = find_ndcg_col(downstream_df, col_prefix)
            if ndcg_col is None:
                print(f"  Model {display_name}: no NDCG column found, skipping.")
                continue

            # Build aligned arrays: only variants present in both nok and downstream
            downstream_variants = set(downstream_df['variant'].values)
            common_variants = sorted(
                set(nok_multi_k.keys()) & downstream_variants
            )

            # Filter out variants with NaN downstream NDCG
            variant_to_ndcg = {}
            for _, row in downstream_df.iterrows():
                v = row['variant']
                ndcg_val = row[ndcg_col]
                if v in common_variants and pd.notna(ndcg_val):
                    variant_to_ndcg[v] = ndcg_val

            usable_variants = sorted(set(common_variants) & set(variant_to_ndcg.keys()))

            if len(usable_variants) < 3:
                print(f"  Model {display_name}: only {len(usable_variants)} usable variants, skipping.")
                continue

            ndcg_values = np.array([variant_to_ndcg[v] for v in usable_variants])

            for K in K_VALUES:
                nok_values = np.array([nok_multi_k[v][K] for v in usable_variants])
                rho, pval = stats.spearmanr(nok_values, ndcg_values)

                # Bootstrap 95% CI
                rng = np.random.default_rng(42)
                boot_rhos = []
                n = len(nok_values)
                for _ in range(2000):
                    idx = rng.choice(n, size=n, replace=True)
                    br, _ = stats.spearmanr(nok_values[idx], ndcg_values[idx])
                    if not np.isnan(br):
                        boot_rhos.append(br)
                ci_lo = np.percentile(boot_rhos, 2.5)
                ci_hi = np.percentile(boot_rhos, 97.5)

                all_rows.append({
                    'dataset': dataset_name,
                    'model': display_name,
                    'K': K,
                    'spearman_rho': rho,
                    'p_value': pval,
                    'n_variants': len(usable_variants),
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                })

                sig = '*' if pval < 0.05 else ''
                print(f"  {display_name} K={K:>2}: rho={rho:+.4f} p={pval:.4f} CI=[{ci_lo:+.2f},{ci_hi:+.2f}] n={len(usable_variants)} {sig}")

    # Save CSV
    results_df = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'k_sensitivity.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")

    # Generate figure
    generate_figure(results_df)

    return results_df


def generate_figure(results_df):
    """Generate the K sensitivity figure."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Style settings matching existing figures
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    colors = {
        'AutoInt': '#E74C3C',
        'DCNv2': '#3498DB',
        'KAR': '#2ECC71',
        'LLM-Rec': '#9B59B6',
        'UniSRec': '#F39C12',
    }
    markers = {
        'AutoInt': 'o',
        'DCNv2': 's',
        'KAR': '^',
        'LLM-Rec': 'D',
        'UniSRec': 'v',
    }

    dataset_display = {
        'amazon-books': 'Amazon-Books',
        'amazon-movies': 'Amazon-Movies',
    }

    datasets = ['amazon-books', 'amazon-movies']
    model_order = ['AutoInt', 'DCNv2', 'KAR', 'LLM-Rec', 'UniSRec']

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_df = results_df[results_df['dataset'] == ds]

        for model in model_order:
            mdf = ds_df[ds_df['model'] == model].sort_values('K')
            if mdf.empty:
                continue

            x = mdf['K'].values
            y = mdf['spearman_rho'].values
            pvals = mdf['p_value'].values

            has_ci = 'ci_lo' in mdf.columns and 'ci_hi' in mdf.columns
            if has_ci:
                ci_lo = mdf['ci_lo'].values
                ci_hi = mdf['ci_hi'].values
                yerr = np.array([y - ci_lo, ci_hi - y])
                ax.errorbar(x, y, yerr=yerr,
                            label=model,
                            color=colors[model],
                            marker=markers[model],
                            markersize=4.5,
                            linewidth=1.2,
                            capsize=2.5,
                            capthick=0.8,
                            elinewidth=0.8,
                            alpha=0.85)
            else:
                ax.plot(x, y,
                        label=model,
                        color=colors[model],
                        marker=markers[model],
                        markersize=5,
                        linewidth=1.3,
                        alpha=0.85)

            # Mark significant points with black edge
            for xi, yi, pi in zip(x, y, pvals):
                if pi < 0.05:
                    ax.plot(xi, yi, marker=markers[model], color=colors[model],
                            markersize=5, markeredgecolor='black',
                            markeredgewidth=0.8, zorder=5)

        ax.set_xlabel('$K$', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Spearman $\\rho$', fontsize=10)
        ax.set_title(dataset_display.get(ds, ds), fontsize=11, fontweight='bold')

        # Use categorical-like x axis with log spacing
        ax.set_xscale('log')
        ax.set_xticks(K_VALUES)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.minorticks_off()

        if idx == 1:
            ax.legend(loc='best', fontsize=7, framealpha=0.9,
                      edgecolor='#cccccc', ncol=1)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'k_sensitivity.pdf')
    fig.savefig(fig_path, format='pdf')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


if __name__ == '__main__':
    run_analysis()
