#!/usr/bin/env python3
"""Generate the redundancy test figure for the ProxyTag paper.

This figure tests whether NO@K discriminates among beneficial tags
(those that improve over the no-tag baseline), not just between
useful and useless ones.

4 subplots: 2 models (AutoInt, DCNv2) x 2 datasets (Amazon-Books, Amazon-Movies)
- Green points: variants above the no-tag baseline
- Red points: variants below the no-tag baseline
- Dashed line: no-tag baseline
- Spearman rho for all points and for above-baseline ("useful") subset only
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Global style settings matching existing paper figures
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

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

# Load data
books_df = pd.read_csv(f'{RESULTS_DIR}/amazon-books_nok_vs_downstream.csv')
movies_df = pd.read_csv(f'{RESULTS_DIR}/amazon-movies_nok_vs_downstream.csv')

# No-tag baselines (from results.md)
baselines = {
    ('AutoInt', 'Amazon-Books'): 0.0655,
    ('DCNv2', 'Amazon-Books'): 0.0416,
    ('AutoInt', 'Amazon-Movies'): 0.0527,
    ('DCNv2', 'Amazon-Movies'): 0.0527,
}

# Define subplots: (model, dataset, dataframe, ndcg_col)
configs = [
    ('AutoInt', 'Amazon-Books', books_df, 'AutoInt_ndcg@10'),
    ('DCNv2', 'Amazon-Books', books_df, 'DCNv2_ndcg@10'),
    ('AutoInt', 'Amazon-Movies', movies_df, 'AutoInt_ndcg@10'),
    ('DCNv2', 'Amazon-Movies', movies_df, 'DCNv2_ndcg@10'),
]

fig, axes = plt.subplots(1, 4, figsize=(7, 2.2), sharey=False)

for ax, (model, dataset, df, ndcg_col) in zip(axes, configs):
    baseline = baselines[(model, dataset)]

    # Drop rows with missing NDCG values
    valid = df[['NO@K', ndcg_col]].dropna()
    nok = valid['NO@K'].values
    ndcg = valid[ndcg_col].values

    # Classify points
    above = ndcg >= baseline
    below = ndcg < baseline

    # Plot points
    ax.scatter(nok[below], ndcg[below], c='#E74C3C', marker='x', s=18, alpha=0.65,
               linewidths=0.8, zorder=3, label='Below baseline')
    ax.scatter(nok[above], ndcg[above], c='#27AE60', marker='o', s=18, alpha=0.65,
               edgecolors='#1E8449', linewidths=0.4, zorder=3, label='Above baseline')

    # Draw baseline
    ax.axhline(y=baseline, color='#7F8C8D', linestyle='--', linewidth=0.9, alpha=0.8, zorder=2)

    # Compute correlations
    rho_full, p_full = stats.spearmanr(nok, ndcg)

    # Within-useful (above baseline) correlation
    if above.sum() >= 3:
        rho_above, p_above = stats.spearmanr(nok[above], ndcg[above])
    else:
        rho_above, p_above = float('nan'), float('nan')

    n_above = above.sum()
    n_total = len(nok)

    # Add correlation text
    text_lines = []
    text_lines.append(r'$\rho_{\mathrm{all}}$' + f' = {rho_full:.2f}' + f' (n={n_total})')
    if not np.isnan(rho_above):
        sig_marker = '*' if p_above < 0.05 else ''
        text_lines.append(r'$\rho_{\mathrm{useful}}$' + f' = {rho_above:.2f}{sig_marker}' + f' (n={n_above})')

    # Position text in upper left
    ax.text(0.04, 0.96, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=6.5, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#BDC3C7', alpha=0.9))

    ax.set_title(f'{model}\n{dataset}', fontsize=8.5, fontweight='bold')
    ax.set_xlabel('NO@K', fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel('NDCG@10', fontsize=8)

    # Add baseline label on right side
    ax.text(0.97, baseline, 'no tags', transform=ax.get_yaxis_transform(),
            fontsize=5.5, color='#7F8C8D', ha='right', va='bottom')

# Shared legend at top
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
               markeredgecolor='#1E8449', markersize=5, label='Above no-tag baseline'),
    plt.Line2D([0], [0], marker='x', color='#E74C3C', markersize=5,
               linestyle='None', markeredgewidth=0.8, label='Below no-tag baseline'),
    plt.Line2D([0], [0], color='#7F8C8D', linestyle='--', linewidth=0.9, label='No-tag baseline'),
]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=7,
           bbox_to_anchor=(0.5, 1.08), frameon=False)

plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/redundancy_test.pdf', bbox_inches='tight')
print("Saved redundancy_test.pdf")

# Print summary statistics
print("\n=== Redundancy Test Summary ===")
for model, dataset, df, ndcg_col in configs:
    baseline = baselines[(model, dataset)]
    valid = df[['NO@K', ndcg_col]].dropna()
    nok = valid['NO@K'].values
    ndcg = valid[ndcg_col].values
    above = ndcg >= baseline

    rho_full, p_full = stats.spearmanr(nok, ndcg)
    n_above = above.sum()
    n_total = len(nok)

    print(f"\n{model} / {dataset}:")
    print(f"  Total variants: {n_total}, Above baseline: {n_above}, Below: {n_total - n_above}")
    print(f"  Full Spearman rho = {rho_full:.4f} (p={p_full:.4e})")

    if n_above >= 3:
        rho_above, p_above = stats.spearmanr(nok[above], ndcg[above])
        print(f"  Within-useful Spearman rho = {rho_above:.4f} (p={p_above:.4e})")
    else:
        print(f"  Too few above-baseline points ({n_above}) for within-useful correlation")
