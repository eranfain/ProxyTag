#!/usr/bin/env python3
"""
Correlation analysis between NO@K (proxy metric) and downstream performance (NDCG@10, Recall@10).

For each tag variant:
1. Computes NO@K using proxy_metric.py
2. Collects NDCG@10 and Recall@10 from existing results JSONs
3. Computes Spearman correlation with significance tests

Produces a table and saves to CSV.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "amazon-books": {
        "train_path": "data/amazon-books/interactions/train.parquet",
        "user_id_col": "userId",
        "item_id_col": "asin",
        "tags_dir": "data/tags/amazon-books/full",
    },
    "amazon-movies": {
        "train_path": "data/amazon-movies/interactions/train.parquet",
        "user_id_col": "userId",
        "item_id_col": "asin",
        "tags_dir": "data/tags/amazon-movies/full",
    },
}


def compute_nok_for_all_variants(dataset_name, K=10, sentence_model='all-MiniLM-L6-v2',
                                  results_dir='results', max_tags=16, panel_size=0.05,
                                  cf_method='bpr'):
    """Compute NO@K for all tag variants in a dataset."""
    from proxytag.core.data import load_df, build_id_mappings
    from proxytag.analysis.proxy_metric import (
        sample_panel_items, get_cf_item_emb, compute_tag_embeddings, evaluate_tag_quality,
        build_cooccurrence_matrix
    )

    cfg = DATASETS[dataset_name]
    tag_files = sorted(glob.glob(os.path.join(cfg["tags_dir"], "*.parquet")))

    if not tag_files:
        print(f"  No tag files found in {cfg['tags_dir']}")
        return {}

    # Load training data and build CF embeddings (once for all variants)
    print(f"  Loading training data...")
    train_df = load_df(cfg["train_path"])

    # Build ID mappings
    val_df = train_df.head(100)
    test_df = train_df.head(100)
    train_df_mapped, _, _, user2idx, idx2user, item2idx, idx2item = build_id_mappings(
        train_df, val_df, test_df,
        user_id_col=cfg["user_id_col"],
        item_id_col=cfg["item_id_col"]
    )
    num_items = len(item2idx)

    # Sample panel items
    panel_items, item_bins = sample_panel_items(
        train_df, cfg["item_id_col"], panel_size=panel_size, n_bins=10, seed=42
    )

    # Train CF model (once)
    print(f"  Training CF model ({cf_method.upper()})...")
    cf_emb_dict = get_cf_item_emb(
        train_df, item2idx, cfg["user_id_col"], cfg["item_id_col"],
        cf_method=cf_method,
    )
    emb_dim = next(iter(cf_emb_dict.values())).shape[0]
    cf_embeddings = np.zeros((num_items, emb_dim), dtype=np.float32)
    for item_id, emb in cf_emb_dict.items():
        if item_id in item2idx:
            cf_embeddings[item2idx[item_id]] = emb

    # Build co-occurrence matrix for RNO@K (once for all variants)
    # Map panel items to global indices
    panel_indices = []
    for item_id in panel_items:
        if item_id in item2idx:
            panel_indices.append(item2idx[item_id])
    panel_indices = np.array(panel_indices)

    print(f"  Building co-occurrence matrix for RNO@K...")
    cooccur_matrix = build_cooccurrence_matrix(
        train_df, cfg["user_id_col"], cfg["item_id_col"],
        item2idx, panel_indices, min_cooccur=3
    )

    # Build sparse user-item matrix for R² proxy
    from scipy.sparse import csr_matrix
    user_map = {u: i for i, u in enumerate(train_df[cfg["user_id_col"]].unique())}
    ui_rows = train_df[cfg["user_id_col"]].map(user_map).values
    ui_cols = train_df[cfg["item_id_col"]].map(item2idx).values
    user_item_matrix = csr_matrix(
        (np.ones(len(train_df)), (ui_rows, ui_cols)),
        shape=(len(user_map), num_items)
    )
    print(f"  User-item matrix: {user_item_matrix.shape}")

    # Compute NO@K, RNO@K, DCG_comp, and R² for each tag variant
    nok_results = {}
    rnok_results = {}
    dcg_comp_results = {}
    r2_results = {}
    for tag_file in tag_files:
        variant = os.path.splitext(os.path.basename(tag_file))[0]
        print(f"\n  Computing NO@{K} + RNO@{K} for: {variant}")

        items_df = pd.read_parquet(tag_file)

        tag_embeddings, tag_item2idx = compute_tag_embeddings(
            items_df, cfg["item_id_col"], 'tags',
            sentence_model, max_tags,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Align tag embeddings
        tag_dim = tag_embeddings.shape[1]
        aligned_tag_embeddings = np.zeros((num_items, tag_dim), dtype=np.float32)
        for item_id, tag_idx in tag_item2idx.items():
            if item_id in item2idx:
                aligned_tag_embeddings[item2idx[item_id]] = tag_embeddings[tag_idx]

        results = evaluate_tag_quality(
            panel_items, item_bins, cf_embeddings, aligned_tag_embeddings,
            item2idx, K=K, n_bins=10,
            cooccur_matrix=cooccur_matrix, cooccur_panel_indices=panel_indices,
            user_item_matrix=user_item_matrix,
        )

        nok_results[variant] = results['overall']['mean_recall']
        if 'rnok_overall' in results:
            rnok_results[variant] = results['rnok_overall']['mean_rnok']
        if 'dcg_comp_overall' in results:
            dcg_comp_results[variant] = results['dcg_comp_overall']['mean_dcg_comp']
        if 'r2_proxy' in results:
            r2_results[variant] = results['r2_proxy']['r2_incremental']

    return nok_results, rnok_results, dcg_comp_results, r2_results


def compute_naive_proxies(dataset_name, panel_items=None):
    """
    Compute naive proxy metrics for each tag variant.
    These serve as baselines to show NO@K is a better predictor than simple statistics.

    Args:
        dataset_name: Key into DATASETS config
        panel_items: Optional set/array of item IDs to restrict computation to (same panel as NO@K)

    Returns dict: variant -> {metric_name: value}
    """
    cfg = DATASETS[dataset_name]
    tag_files = sorted(glob.glob(os.path.join(cfg["tags_dir"], "*.parquet")))
    item_id_col = cfg["item_id_col"]

    if panel_items is not None:
        panel_set = set(panel_items)
        print(f"  Restricting naive proxies to panel of {len(panel_set)} items")
    else:
        panel_set = None

    results = {}
    for tag_file in tag_files:
        variant = os.path.splitext(os.path.basename(tag_file))[0]
        df = pd.read_parquet(tag_file)
        if panel_set is not None:
            df = df[df[item_id_col].isin(panel_set)]
        tags_col = df['tags'].fillna('')

        # 1. Tag Coverage: fraction of items that have at least one tag
        has_tags = (tags_col.str.strip() != '').sum()
        coverage = has_tags / len(df)

        # 2. Avg Tag Count: mean number of tags per item
        def _split_tags(x):
            if not x:
                return []
            return [t.strip() for t in (x.split('|') if '|' in x else x.split(',')) if t.strip()]
        tag_counts = tags_col.apply(lambda x: len(_split_tags(x)))
        avg_count = tag_counts.mean()

        # 3. Tag Diversity: number of unique tags across all items (normalized by total tags)
        all_tags = []
        for t in tags_col:
            if t:
                all_tags.extend([tag.strip().lower() for tag in _split_tags(t)])
        n_unique = len(set(all_tags))
        diversity = n_unique / max(len(all_tags), 1)

        # 4. Avg Tag Length: average character length of tags (proxy for specificity)
        if all_tags:
            avg_length = np.mean([len(t) for t in all_tags])
        else:
            avg_length = 0

        results[variant] = {
            'avg_tag_count': avg_count,
            'tag_diversity': diversity,
            'avg_tag_length': avg_length,
            'semantic_diversity': None,
            'inter_item_similarity': None,
            'tag_interaction_alignment': None,
        }

    # Embedding-based proxies using disk-backed tag embedding cache
    print(f"  Computing embedding-based proxies with sentence-transformers...")
    from sentence_transformers import SentenceTransformer
    from proxytag.core.tags import _clean_tag_string
    from proxytag.core.data import load_df

    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    emb_dim = st_model.get_sentence_embedding_dimension()

    # Load training data for tag-interaction alignment
    train_df = load_df(cfg["train_path"])

    # Build or load disk-backed tag embedding cache (memmap + index)
    cache_dir = os.path.join(os.path.dirname(cfg["tags_dir"]), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    tag_index_path = os.path.join(cache_dir, 'tag_index.json')
    tag_emb_path = os.path.join(cache_dir, 'tag_embeddings.npy')

    # Collect all unique tags
    print(f"  Collecting unique tags...")
    item_id_col = cfg["item_id_col"]
    global_tag_set = set()
    for tag_file in tag_files:
        df = pd.read_parquet(tag_file)
        if panel_set is not None:
            df = df[df[item_id_col].isin(panel_set)]
        for tag_str in df['tags'].fillna(''):
            if tag_str and tag_str.strip():
                cleaned = _clean_tag_string(tag_str)
                for t in cleaned.split('|'):
                    t = t.strip()
                    if t:
                        global_tag_set.add(t)

    unique_tags = sorted(global_tag_set)
    print(f"  {len(unique_tags)} unique tags found")

    # Check if cache is valid (same tags)
    cache_valid = False
    if os.path.exists(tag_index_path) and os.path.exists(tag_emb_path):
        with open(tag_index_path) as f:
            cached_index = json.load(f)
        if set(cached_index.keys()) == global_tag_set:
            cache_valid = True
            print(f"  Loading tag embeddings from disk cache...")
            tag_to_idx = {t: int(i) for t, i in cached_index.items()}
        else:
            print(f"  Cache stale ({len(cached_index)} cached vs {len(unique_tags)} current), re-encoding...")

    if not cache_valid and unique_tags:
        # Encode in chunks to limit peak RAM
        CHUNK_SIZE = 50000
        tag_to_idx = {t: i for i, t in enumerate(unique_tags)}

        # Create memmap file on disk
        mmap_emb = np.memmap(tag_emb_path, dtype='float32', mode='w+',
                             shape=(len(unique_tags), emb_dim))

        for chunk_start in range(0, len(unique_tags), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(unique_tags))
            chunk_tags = unique_tags[chunk_start:chunk_end]
            print(f"  Encoding tags {chunk_start}–{chunk_end} / {len(unique_tags)}...")
            chunk_embs = st_model.encode(chunk_tags, batch_size=512,
                                          show_progress_bar=False, normalize_embeddings=True)
            mmap_emb[chunk_start:chunk_end] = chunk_embs
            del chunk_embs

        mmap_emb.flush()
        del mmap_emb

        # Save index
        with open(tag_index_path, 'w') as f:
            json.dump(tag_to_idx, f)
        print(f"  Saved tag embedding cache to {cache_dir}")

    # Open as read-only memmap (OS pages in on demand, doesn't load into RAM)
    if unique_tags:
        tag_embeddings_mmap = np.memmap(tag_emb_path, dtype='float32', mode='r',
                                         shape=(len(unique_tags), emb_dim))
    else:
        tag_embeddings_mmap = np.zeros((0, emb_dim), dtype=np.float32)
        tag_to_idx = {}

    def get_tag_vectors(tags):
        """Look up tag embeddings from memmap. Returns (n_found, emb_dim) array."""
        idxs = [tag_to_idx[t] for t in tags if t in tag_to_idx]
        if not idxs:
            return np.zeros((0, emb_dim), dtype=np.float32)
        return np.array(tag_embeddings_mmap[idxs])

    for tag_file in tag_files:
        variant = os.path.splitext(os.path.basename(tag_file))[0]
        df = pd.read_parquet(tag_file)
        if panel_set is not None:
            df = df[df[item_id_col].isin(panel_set)]
        tags_col = df['tags'].fillna('')

        item_ids = df[item_id_col].values

        # Build per-item tag lists and item embedding matrix
        item_emb_matrix = np.zeros((len(df), emb_dim), dtype=np.float32)
        non_empty_mask = []
        item_tag_lists = []

        for i, tag_str in enumerate(tags_col):
            if tag_str and tag_str.strip():
                cleaned = _clean_tag_string(tag_str)
                tags = [t.strip() for t in cleaned.split('|') if t.strip()][:16]
                if tags:
                    non_empty_mask.append(True)
                    item_tag_lists.append(tags)
                    vecs = get_tag_vectors(tags)
                    if len(vecs) > 0:
                        item_emb_matrix[i] = np.mean(vecs, axis=0)
                        item_emb_matrix[i] /= np.linalg.norm(item_emb_matrix[i]) + 1e-8
                else:
                    non_empty_mask.append(False)
                    item_tag_lists.append([])
            else:
                non_empty_mask.append(False)
                item_tag_lists.append([])

        non_empty_mask = np.array(non_empty_mask)

        # --- Proxy 5: Semantic Diversity ---
        # Avg per-item std of individual tag embeddings
        item_stds = []
        for tags in item_tag_lists:
            if len(tags) < 2:
                continue
            vecs = get_tag_vectors(tags)
            if len(vecs) < 2:
                continue
            item_stds.append(np.std(vecs, axis=0).mean())

        results[variant]['semantic_diversity'] = float(np.mean(item_stds)) if item_stds else 0.0

        # --- Proxy 6: Inter-item Similarity ---
        valid_embs = item_emb_matrix[non_empty_mask]
        if len(valid_embs) > 100:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(valid_embs), size=min(500, len(valid_embs)), replace=False)
            sample_embs = valid_embs[sample_idx]
            sim_matrix = sample_embs @ sample_embs.T
            np.fill_diagonal(sim_matrix, 0)
            avg_sim = sim_matrix.sum() / (len(sample_idx) * (len(sample_idx) - 1))
            results[variant]['inter_item_similarity'] = float(avg_sim)
        else:
            results[variant]['inter_item_similarity'] = 0.0

        # --- Proxy 7: Tag-Interaction Alignment ---
        item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        user_alignments = []
        user_items = train_df.groupby(cfg["user_id_col"])[item_id_col].apply(list)

        sample_users = user_items.sample(n=min(500, len(user_items)), random_state=42)
        for user_item_list in sample_users:
            idxs = [item_id_to_idx[iid] for iid in user_item_list if iid in item_id_to_idx]
            if len(idxs) < 2:
                continue
            user_embs = item_emb_matrix[idxs]
            valid = np.linalg.norm(user_embs, axis=1) > 0
            user_embs = user_embs[valid]
            if len(user_embs) < 2:
                continue
            sim = user_embs @ user_embs.T
            np.fill_diagonal(sim, 0)
            avg_user_sim = sim.sum() / (len(user_embs) * (len(user_embs) - 1))
            user_alignments.append(avg_user_sim)

        results[variant]['tag_interaction_alignment'] = float(np.mean(user_alignments)) if user_alignments else 0.0

        print(f"    {variant}: sem_div={results[variant]['semantic_diversity']:.4f}, "
              f"inter_sim={results[variant]['inter_item_similarity']:.4f}, "
              f"align={results[variant]['tag_interaction_alignment']:.4f}")

    return results


def collect_downstream_metrics(dataset_name, results_dir='results'):
    """Collect NDCG@10 and Recall@10 from existing result JSONs for a dataset."""
    pattern = os.path.join(results_dir, f"*_{dataset_name}_results.json")
    result_files = glob.glob(pattern)

    metrics_by_variant = {}
    for path in result_files:
        with open(path) as f:
            data = json.load(f)

        job_name = data.get('job_name', '')

        # Skip no-tags baselines
        if 'no_tags' in job_name:
            continue

        # Extract variant name (remove model prefix and dataset suffix)
        # Formats: {model}_{variant}_{dataset}, hybrid_{variant}_{dataset}
        variant = job_name.replace(f'_{dataset_name}', '')
        for prefix in ['dcnv2_', 'autoint_', 'unisrec_', 'kar_', 'llmrec_']:
            if variant.startswith(prefix):
                variant = variant[len(prefix):]
                break

        # Get metrics
        test_m = data.get('test_metrics', {})
        ndcg = data.get('ndcg@10', test_m.get('ndcg@10', None))
        recall = data.get('recall@10', test_m.get('recall@10', None))
        ndcg_std = test_m.get('ndcg@10_std', None)
        n_eval = test_m.get('n_eval', data.get('n_eval', None))
        model = data.get('model', job_name.split('_')[0])

        if ndcg is not None:
            key = (variant, model)
            metrics_by_variant[key] = {
                'ndcg@10': ndcg,
                'recall@10': recall,
                'ndcg@10_std': ndcg_std,
                'n_eval': n_eval,
                'model': model,
            }

    return metrics_by_variant


def bootstrap_spearman_ci(x, y, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for Spearman correlation."""
    rng = np.random.default_rng(seed)
    n = len(x)
    x, y = np.array(x), np.array(y)
    rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            rhos.append(rho)
    alpha = (1 - ci) / 2
    lo = np.percentile(rhos, alpha * 100)
    hi = np.percentile(rhos, (1 - alpha) * 100)
    return lo, hi


def _welch_z_significant(m1, s1, n1, m2, s2, n2, alpha=0.05):
    """Two-sided Welch's t-test using stored mean/std/n. Returns True if significant."""
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        return False
    z = abs(m1 - m2) / se
    p = 2 * (1 - stats.norm.cdf(z))
    return p < alpha


def _compute_concordance(proxy_values, ndcg_values, ndcg_stds=None, n_evals=None,
                         sig_alpha=0.05):
    """Compute pair-level concordance between proxy and NDCG.

    When ndcg_stds and n_evals are provided, "significant pairs" are those where
    the two variants have significantly different NDCG (Welch's z-test on per-user
    means). Otherwise all non-tied pairs are used for the filtered concordance.
    """
    n = len(proxy_values)
    proxy_values = np.array(proxy_values)
    ndcg_values = np.array(ndcg_values)
    has_stats = ndcg_stds is not None and n_evals is not None

    concordant = 0
    discordant = 0
    tied = 0
    concordant_sig = 0
    discordant_sig = 0
    n_sig = 0

    for i in range(n):
        for j in range(i + 1, n):
            dp = proxy_values[i] - proxy_values[j]
            dn = ndcg_values[i] - ndcg_values[j]

            if dp * dn > 0:
                concordant += 1
            elif dp * dn < 0:
                discordant += 1
            else:
                tied += 1

            if has_stats:
                is_sig = _welch_z_significant(
                    ndcg_values[i], ndcg_stds[i], n_evals[i],
                    ndcg_values[j], ndcg_stds[j], n_evals[j],
                    alpha=sig_alpha,
                )
            else:
                is_sig = abs(dn) > 0

            if is_sig:
                n_sig += 1
                if dp * dn > 0:
                    concordant_sig += 1
                elif dp * dn < 0:
                    discordant_sig += 1

    total = concordant + discordant + tied
    conc_rate = (concordant + 0.5 * tied) / total if total > 0 else 0.5

    if n_sig > 0:
        decided_sig = concordant_sig + discordant_sig
        conc_rate_sig = concordant_sig / decided_sig if decided_sig > 0 else float('nan')
    else:
        conc_rate_sig = float('nan')

    # Binomial test on all decided pairs
    n_decided = concordant + discordant
    if n_decided > 0:
        p_binom = stats.binomtest(concordant, n_decided, 0.5, alternative='greater').pvalue
    else:
        p_binom = 1.0

    # Binomial test on significant pairs only
    decided_sig = concordant_sig + discordant_sig
    if decided_sig > 0:
        p_binom_sig = stats.binomtest(concordant_sig, decided_sig, 0.5, alternative='greater').pvalue
    else:
        p_binom_sig = 1.0

    # Kendall's tau
    tau, p_tau = stats.kendalltau(proxy_values, ndcg_values)

    # Bootstrap CI on concordance rate
    rng = np.random.default_rng(42)
    boot_concs = []
    for _ in range(2000):
        idx = rng.choice(n, size=n, replace=True)
        bp = proxy_values[idx]
        bn = ndcg_values[idx]
        bc, bd, bt = 0, 0, 0
        for ii in range(n):
            for jj in range(ii + 1, n):
                dp = bp[ii] - bp[jj]
                dn = bn[ii] - bn[jj]
                if dp * dn > 0:
                    bc += 1
                elif dp * dn < 0:
                    bd += 1
                else:
                    bt += 1
        bt_total = bc + bd + bt
        boot_concs.append((bc + 0.5 * bt) / bt_total if bt_total > 0 else 0.5)

    ci_lo = np.percentile(boot_concs, 2.5)
    ci_hi = np.percentile(boot_concs, 97.5)

    return {
        'concordance': conc_rate,
        'concordance_sig': conc_rate_sig,
        'n_pairs': total,
        'n_sig_pairs': n_sig,
        'n_concordant': concordant,
        'n_discordant': discordant,
        'n_concordant_sig': concordant_sig,
        'n_discordant_sig': discordant_sig,
        'p_value_binomial': p_binom,
        'p_value_binomial_sig': p_binom_sig,
        'kendall_tau': tau,
        'kendall_p': p_tau,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
    }


def run_concordance_analysis(nok_results, downstream_metrics, dataset_name,
                             naive_proxies=None, rnok_results=None,
                             dcg_comp_results=None, r2_results=None):
    """Pair-level concordance: does higher proxy → higher NDCG@10?

    For pairs where the two variants have significantly different NDCG
    (Welch's z-test on per-user means using stored std/n_eval), reports
    concordance_sig — the decision accuracy on statistically distinguishable pairs.
    """
    rows = []
    models = set(m for (_, m) in downstream_metrics.keys())

    for model in sorted(models):
        model_metrics = {v: m for (v, mod), m in downstream_metrics.items() if mod == model}
        common_variants = set(nok_results.keys()) & set(model_metrics.keys())
        if naive_proxies:
            common_variants = common_variants & set(naive_proxies.keys())
        if len(common_variants) < 3:
            continue

        variants = sorted(common_variants)
        ndcg_values = [model_metrics[v]['ndcg@10'] for v in variants]
        ndcg_stds = [model_metrics[v].get('ndcg@10_std') for v in variants]
        n_evals = [model_metrics[v].get('n_eval') for v in variants]

        has_stats = all(s is not None for s in ndcg_stds) and all(n is not None for n in n_evals)
        if has_stats:
            ndcg_stds = np.array(ndcg_stds, dtype=float)
            n_evals = np.array(n_evals, dtype=float)
        else:
            ndcg_stds = None
            n_evals = None

        n_pairs = len(variants) * (len(variants) - 1) // 2
        print(f"\n  {model} ({len(variants)} variants, {n_pairs} pairs):")
        header = f"    {'Proxy':<25} {'conc':>6} {'conc_sig':>8} {'n_sig':>5} {'p_all':>8} {'p_sig':>8} {'tau':>6} {'p_tau':>8} {'95% CI':>16} {'sig':>4}"
        print(header)
        print(f"    {'-'*len(header)}")

        def _log_conc(proxy_name, proxy_vals, ndcg_vals, ndcg_s, n_ev):
            r = _compute_concordance(proxy_vals, ndcg_vals, ndcg_s, n_ev)
            sig = '*' if r['p_value_binomial'] < 0.05 else ''
            conc_sig_str = f"{r['concordance_sig']:.3f}" if not np.isnan(r['concordance_sig']) else 'n/a'
            print(f"    {proxy_name:<25} {r['concordance']:>6.3f} {conc_sig_str:>8} {r['n_sig_pairs']:>5} "
                  f"{r['p_value_binomial']:>8.4f} {r['p_value_binomial_sig']:>8.4f} {r['kendall_tau']:>+6.3f} {r['kendall_p']:>8.4f} "
                  f"[{r['ci_lo']:>+6.3f}, {r['ci_hi']:>+6.3f}] {sig}")
            rows.append({
                'dataset': dataset_name,
                'model': model,
                'n_variants': len(variants),
                'n_pairs': r['n_pairs'],
                'proxy': proxy_name,
                'concordance': r['concordance'],
                'concordance_sig': r['concordance_sig'],
                'n_sig_pairs': r['n_sig_pairs'],
                'p_value_binomial': r['p_value_binomial'],
                'p_value_binomial_sig': r['p_value_binomial_sig'],
                'kendall_tau': r['kendall_tau'],
                'kendall_p': r['kendall_p'],
                'ci_lo': r['ci_lo'],
                'ci_hi': r['ci_hi'],
            })

        _log_conc('NO@K', [nok_results[v] for v in variants], ndcg_values, ndcg_stds, n_evals)

        if rnok_results:
            rnok_variants = [v for v in variants if v in rnok_results]
            if len(rnok_variants) >= 3:
                rv_ndcg = [model_metrics[v]['ndcg@10'] for v in rnok_variants]
                rv_std = np.array([model_metrics[v].get('ndcg@10_std', 0) for v in rnok_variants]) if has_stats else None
                rv_n = np.array([model_metrics[v].get('n_eval', 0) for v in rnok_variants]) if has_stats else None
                _log_conc('RNO@K', [rnok_results[v] for v in rnok_variants], rv_ndcg, rv_std, rv_n)

        if dcg_comp_results:
            dcg_variants = [v for v in variants if v in dcg_comp_results]
            if len(dcg_variants) >= 3:
                dv_ndcg = [model_metrics[v]['ndcg@10'] for v in dcg_variants]
                dv_std = np.array([model_metrics[v].get('ndcg@10_std', 0) for v in dcg_variants]) if has_stats else None
                dv_n = np.array([model_metrics[v].get('n_eval', 0) for v in dcg_variants]) if has_stats else None
                _log_conc('DCG_comp', [dcg_comp_results[v] for v in dcg_variants], dv_ndcg, dv_std, dv_n)

        if r2_results:
            r2_variants = [v for v in variants if v in r2_results]
            if len(r2_variants) >= 3:
                r2v_ndcg = [model_metrics[v]['ndcg@10'] for v in r2_variants]
                r2v_std = np.array([model_metrics[v].get('ndcg@10_std', 0) for v in r2_variants]) if has_stats else None
                r2v_n = np.array([model_metrics[v].get('n_eval', 0) for v in r2_variants]) if has_stats else None
                _log_conc('R2_incremental', [r2_results[v] for v in r2_variants], r2v_ndcg, r2v_std, r2v_n)

        if naive_proxies:
            for proxy_name in ['avg_tag_count', 'tag_diversity', 'avg_tag_length',
                               'semantic_diversity', 'inter_item_similarity', 'tag_interaction_alignment']:
                proxy_values = [naive_proxies[v][proxy_name] for v in variants]
                _log_conc(proxy_name, proxy_values, ndcg_values, ndcg_stds, n_evals)

    return pd.DataFrame(rows)


def run_correlation_analysis(nok_results, downstream_metrics, dataset_name, naive_proxies=None, rnok_results=None, dcg_comp_results=None, r2_results=None):
    """Compute Spearman correlations between proxy metrics and downstream metrics."""
    rows = []

    # Group by model
    models = set(m for (_, m) in downstream_metrics.keys())

    for model in sorted(models):
        model_metrics = {v: m for (v, mod), m in downstream_metrics.items() if mod == model}

        common_variants = set(nok_results.keys()) & set(model_metrics.keys())
        if naive_proxies:
            common_variants = common_variants & set(naive_proxies.keys())
        if len(common_variants) < 3:
            print(f"  {model}: Only {len(common_variants)} common variants, skipping (need >= 3)")
            continue

        variants = sorted(common_variants)
        ndcg_values = [model_metrics[v]['ndcg@10'] for v in variants]

        print(f"\n  {model} ({len(variants)} variants):")
        print(f"    {'Proxy Metric':<25} {'spearman':>8} {'pearson':>8} {'p(sp)':>8} {'p(pe)':>8} {'95% CI (sp)':>16} {'sig':>4}")
        print(f"    {'-'*80}")

        # NO@K correlation
        nok_values = [nok_results[v] for v in variants]
        rho_sp, p_sp = stats.spearmanr(nok_values, ndcg_values)
        rho_pe, p_pe = stats.pearsonr(nok_values, ndcg_values)
        ci_lo, ci_hi = bootstrap_spearman_ci(nok_values, ndcg_values)
        print(f"    {'NO@K':<25} {rho_sp:>+8.4f} {rho_pe:>+8.4f} {p_sp:>8.4f} {p_pe:>8.4f} [{ci_lo:>+6.3f}, {ci_hi:>+6.3f}] {'*' if p_sp < 0.05 else ''}")

        rows.append({
            'dataset': dataset_name,
            'model': model,
            'n_variants': len(variants),
            'proxy': 'NO@K',
            'spearman_rho': rho_sp,
            'pearson_r': rho_pe,
            'p_value': p_sp,
            'p_value_pearson': p_pe,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
        })

        # Helper to compute and log correlation with CI
        def _log_proxy(proxy_name, proxy_vals, ndcg_vals, n_var):
            rho_sp, p_sp = stats.spearmanr(proxy_vals, ndcg_vals)
            rho_pe, p_pe = stats.pearsonr(proxy_vals, ndcg_vals)
            ci_lo, ci_hi = bootstrap_spearman_ci(proxy_vals, ndcg_vals)
            print(f"    {proxy_name:<25} {rho_sp:>+8.4f} {rho_pe:>+8.4f} {p_sp:>8.4f} {p_pe:>8.4f} [{ci_lo:>+6.3f}, {ci_hi:>+6.3f}] {'*' if p_sp < 0.05 else ''}")
            rows.append({
                'dataset': dataset_name,
                'model': model,
                'n_variants': n_var,
                'proxy': proxy_name,
                'spearman_rho': rho_sp,
                'pearson_r': rho_pe,
                'p_value': p_sp,
                'p_value_pearson': p_pe,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
            })

        # RNO@K correlation
        if rnok_results:
            rnok_variants = [v for v in variants if v in rnok_results]
            if len(rnok_variants) >= 3:
                _log_proxy('RNO@K',
                           [rnok_results[v] for v in rnok_variants],
                           [model_metrics[v]['ndcg@10'] for v in rnok_variants],
                           len(rnok_variants))

        # DCG_comp correlation
        if dcg_comp_results:
            dcg_variants = [v for v in variants if v in dcg_comp_results]
            if len(dcg_variants) >= 3:
                _log_proxy('DCG_comp',
                           [dcg_comp_results[v] for v in dcg_variants],
                           [model_metrics[v]['ndcg@10'] for v in dcg_variants],
                           len(dcg_variants))

        # R² proxy correlation
        if r2_results:
            r2_variants = [v for v in variants if v in r2_results]
            if len(r2_variants) >= 3:
                _log_proxy('R2_incremental',
                           [r2_results[v] for v in r2_variants],
                           [model_metrics[v]['ndcg@10'] for v in r2_variants],
                           len(r2_variants))

        # Naive proxy correlations
        if naive_proxies:
            proxy_names = ['avg_tag_count', 'tag_diversity', 'avg_tag_length',
                              'semantic_diversity', 'inter_item_similarity', 'tag_interaction_alignment']
            for proxy_name in proxy_names:
                proxy_values = [naive_proxies[v][proxy_name] for v in variants]
                _log_proxy(proxy_name, proxy_values, ndcg_values, len(variants))

    return pd.DataFrame(rows)


def build_full_table(nok_results, downstream_metrics, dataset_name, rnok_results=None):
    """Build a table with NO@K, RNO@K, and all downstream metrics per variant."""
    rows = []
    for variant, nok in sorted(nok_results.items()):
        row = {'variant': variant, 'NO@K': nok}

        if rnok_results and variant in rnok_results:
            row['RNO@K'] = rnok_results[variant]

        # Add downstream metrics for each model
        for (v, model), metrics in downstream_metrics.items():
            if v == variant:
                row[f'{model}_ndcg@10'] = metrics['ndcg@10']
                row[f'{model}_recall@10'] = metrics['recall@10']

        rows.append(row)

    return pd.DataFrame(rows).sort_values('NO@K', ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis: NO@K vs downstream metrics")
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS.keys()),
                        default=list(DATASETS.keys()))
    parser.add_argument('--K', type=int, default=10, help='K for NO@K (default: 10)')
    parser.add_argument('--max_tags', type=int, default=16)
    parser.add_argument('--panel_size', type=float, default=0.05,
                        help='Fraction of items for panel (default: 0.05)')
    parser.add_argument('--sentence_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--skip_nok', action='store_true',
                        help='Skip NO@K computation, load from existing results')
    parser.add_argument('--skip_naive', action='store_true',
                        help='Skip naive proxy computation, load from existing cache')
    parser.add_argument('--cf_method', type=str, default='bpr', choices=['als', 'bpr'],
                        help='CF model for NO@K computation (default: bpr)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in args.datasets:
        print("\n" + "=" * 80)
        print(f"CORRELATION ANALYSIS: {dataset_name}")
        print("=" * 80)

        # Compute or load NO@K and RNO@K
        panel_pct = int(args.panel_size * 100)
        cf_suffix = f'_{args.cf_method}' if args.cf_method != 'als' else ''
        nok_cache_path = os.path.join(args.output_dir, f'{dataset_name}_nok_results_panel{panel_pct}{cf_suffix}.json')
        rnok_cache_path = os.path.join(args.output_dir, f'{dataset_name}_rnok_results_panel{panel_pct}{cf_suffix}.json')

        dcg_cache_path = os.path.join(args.output_dir, f'{dataset_name}_dcg_comp_results_panel{panel_pct}{cf_suffix}.json')
        r2_cache_path = os.path.join(args.output_dir, f'{dataset_name}_r2_results_panel{panel_pct}{cf_suffix}.json')

        if args.skip_nok and os.path.exists(nok_cache_path):
            print(f"  Loading cached NO@K from {nok_cache_path}")
            with open(nok_cache_path) as f:
                nok_results = json.load(f)
            rnok_results = {}
            if os.path.exists(rnok_cache_path):
                print(f"  Loading cached RNO@K from {rnok_cache_path}")
                with open(rnok_cache_path) as f:
                    rnok_results = json.load(f)
            dcg_comp_results = {}
            if os.path.exists(dcg_cache_path):
                print(f"  Loading cached DCG_comp from {dcg_cache_path}")
                with open(dcg_cache_path) as f:
                    dcg_comp_results = json.load(f)
            r2_results = {}
            if os.path.exists(r2_cache_path):
                print(f"  Loading cached R² from {r2_cache_path}")
                with open(r2_cache_path) as f:
                    r2_results = json.load(f)
        else:
            print(f"  Computing proxies for all tag variants (panel_size={args.panel_size}, cf={args.cf_method})...")
            nok_results, rnok_results, dcg_comp_results, r2_results = compute_nok_for_all_variants(
                dataset_name, K=args.K,
                sentence_model=args.sentence_model,
                results_dir=args.results_dir,
                max_tags=args.max_tags,
                panel_size=args.panel_size,
                cf_method=args.cf_method,
            )
            # Cache
            with open(nok_cache_path, 'w') as f:
                json.dump(nok_results, f, indent=2)
            print(f"  Cached NO@K to {nok_cache_path}")
            if rnok_results:
                with open(rnok_cache_path, 'w') as f:
                    json.dump(rnok_results, f, indent=2)
                print(f"  Cached RNO@K to {rnok_cache_path}")
            if dcg_comp_results:
                with open(dcg_cache_path, 'w') as f:
                    json.dump(dcg_comp_results, f, indent=2)
                print(f"  Cached DCG_comp to {dcg_cache_path}")
            if r2_results:
                with open(r2_cache_path, 'w') as f:
                    json.dump(r2_results, f, indent=2)
                print(f"  Cached R² to {r2_cache_path}")

        if not nok_results:
            print(f"  No NO@K results, skipping.")
            continue

        # Sample panel items (deterministic, same panel used for NO@K)
        cfg = DATASETS[dataset_name]
        train_df = pd.read_parquet(cfg["train_path"])
        item_counts = train_df[cfg["item_id_col"]].value_counts()
        n_items = len(item_counts)
        n_bins = 10
        item_bins = {}
        for i, (item_id, count) in enumerate(item_counts.items()):
            item_bins[item_id] = min(i * n_bins // n_items, n_bins - 1)
        np.random.seed(42)
        panel_items = []
        for bin_idx in range(n_bins):
            bin_items = [item for item, b in item_bins.items() if b == bin_idx]
            n_sample = max(1, int(len(bin_items) * args.panel_size))
            sampled = np.random.choice(bin_items, size=n_sample, replace=False)
            panel_items.extend(sampled)
        panel_items = np.array(panel_items)
        print(f"  Panel items: {len(panel_items)} ({len(panel_items)/n_items:.1%} of {n_items})")

        # Compute or load naive proxy baselines (restricted to panel items)
        naive_cache_path = os.path.join(args.output_dir, f'{dataset_name}_naive_proxies_panel{panel_pct}_cache.json')
        if args.skip_naive and os.path.exists(naive_cache_path):
            print(f"\n  Loading cached naive proxies from {naive_cache_path}")
            with open(naive_cache_path) as f:
                naive_proxies = json.load(f)
        else:
            print(f"\n  Computing naive proxy baselines (panel={len(panel_items)} items)...")
            naive_proxies = compute_naive_proxies(dataset_name, panel_items=panel_items)
            with open(naive_cache_path, 'w') as f:
                json.dump(naive_proxies, f, indent=2)
            print(f"  Cached naive proxies to {naive_cache_path}")
        print(f"  {len(naive_proxies)} variants × 7 naive metrics")

        # Collect downstream metrics
        print(f"\n  Collecting downstream metrics from {args.results_dir}...")
        downstream_metrics = collect_downstream_metrics(dataset_name, args.results_dir)
        print(f"  Found {len(downstream_metrics)} (variant, model) pairs")

        # Correlation analysis (NO@K + RNO@K + naive proxies)
        print(f"\n  --- Correlation Results ---")
        corr_df = run_correlation_analysis(nok_results, downstream_metrics, dataset_name, naive_proxies, rnok_results, dcg_comp_results, r2_results)

        if len(corr_df) > 0:
            corr_path = os.path.join(args.output_dir, f'{dataset_name}_correlation.csv')
            corr_df.to_csv(corr_path, index=False)
            print(f"\n  Saved to: {corr_path}")

        # Pair-level concordance analysis
        print(f"\n  --- Concordance Results (pair-level) ---")
        conc_df = run_concordance_analysis(nok_results, downstream_metrics, dataset_name, naive_proxies, rnok_results, dcg_comp_results, r2_results)

        if len(conc_df) > 0:
            conc_path = os.path.join(args.output_dir, f'{dataset_name}_concordance.csv')
            conc_df.to_csv(conc_path, index=False)
            print(f"\n  Saved to: {conc_path}")

        # Full table
        full_table = build_full_table(nok_results, downstream_metrics, dataset_name, rnok_results)
        if len(full_table) > 0:
            print(f"\n  --- Full Table ---")
            print(full_table.to_string(index=False))

            table_path = os.path.join(args.output_dir, f'{dataset_name}_nok_vs_downstream.csv')
            full_table.to_csv(table_path, index=False)
            print(f"\n  Saved to: {table_path}")


if __name__ == '__main__':
    main()
