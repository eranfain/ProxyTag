#!/usr/bin/env python3
"""Generate all publication-quality figures for the ProxyTag paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Global style settings
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
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,
})

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

# ============================================================================
# Figure 1: ProxyTag Overview Flow Diagram
# ============================================================================
def make_figure1():
    fig, ax = plt.subplots(1, 1, figsize=(7, 2.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Color definitions
    c_data = '#D6EAF8'       # light blue for data
    c_compute = '#FDEBD0'    # light orange for computation
    c_output = '#D5F5E3'     # light green for output
    c_border_data = '#5DADE2'
    c_border_compute = '#E59866'
    c_border_output = '#58D68D'

    def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=7, bold=False):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.15",
                             facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=1.2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, wrap=True)

    def draw_arrow(x1, y1, x2, y2, style='->', color='#555555', shrink_a=6, shrink_b=6):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                    lw=1.3, connectionstyle='arc3,rad=0',
                                    shrinkA=shrink_a, shrinkB=shrink_b))

    # Stage 1: LLM Tag Variants (data)
    draw_box(0.2, 1.8, 2.2, 1.8, 'LLM Tag\nVariants\n(V1, V2, ..., V32)',
             c_data, c_border_data, fontsize=7, bold=True)

    # Arrow from variants to panel sampling
    draw_arrow(2.4, 2.7, 3.2, 2.7)

    # Stage 2: Panel Sampling (computation)
    draw_box(3.2, 1.8, 2.2, 1.8, 'Panel\nSampling\n(stratified ~5%)',
             c_compute, c_border_compute, fontsize=7, bold=True)

    # Two parallel paths diverge
    # Arrow to CF Model (upper path)
    draw_arrow(5.4, 3.2, 6.3, 4.1)
    # Arrow to Tag Encoding (lower path)
    draw_arrow(5.4, 2.2, 6.3, 1.3)

    # Stage 3a: CF Model (computation, upper)
    draw_box(6.3, 3.6, 2.4, 1.4, 'CF Model (BPR)\nCF item\nneighborhoods',
             c_compute, c_border_compute, fontsize=7, bold=True)

    # Stage 3b: Tag Encoding (computation, lower)
    draw_box(6.3, 0.5, 2.4, 1.4, 'Sentence-BERT\nEncoding\ntag neighborhoods',
             c_compute, c_border_compute, fontsize=7, bold=True)

    # Arrows converging
    draw_arrow(8.7, 4.0, 9.6, 3.1)
    draw_arrow(8.7, 1.5, 9.6, 2.3)

    # Stage 4: NO@K (output)
    draw_box(9.6, 1.8, 2.4, 1.8, 'Neighborhood\nOverlap\n(NO@K)',
             c_output, c_border_output, fontsize=7.5, bold=True)

    # Arrow to output
    draw_arrow(12.0, 2.7, 12.7, 2.7)

    # Stage 5: Ranked list (output)
    draw_box(12.7, 2.0, 1.1, 1.4, 'Ranked\nVariant\nList',
             c_output, c_border_output, fontsize=7, bold=True)

    fig.savefig(f'{FIGURES_DIR}/proxytag_overview.pdf', format='pdf')
    plt.close(fig)
    print("Figure 1 saved: proxytag_overview.pdf")


# ============================================================================
# Figure 2: NO@K vs Downstream NDCG@10 Scatter Plots
# ============================================================================
def make_figure2():
    books = pd.read_csv(f'{RESULTS_DIR}/amazon-books_nok_vs_downstream.csv')
    movies = pd.read_csv(f'{RESULTS_DIR}/amazon-movies_nok_vs_downstream.csv')

    # Identify model columns (NDCG@10)
    model_names_display = ['AutoInt', 'DCNv2', 'KAR', 'LLM-Rec', 'UniSRec']

    def get_ndcg_col(df, model):
        """Find the ndcg@10 column for a given model."""
        for col in df.columns:
            if model.lower().replace('-', '') in col.lower().replace('-', '').replace('_', '') and 'ndcg' in col.lower():
                return col
        # Try more flexible matching
        if model == 'UniSRec':
            for col in df.columns:
                if 'unisrec' in col.lower() and 'ndcg' in col.lower():
                    return col
        return None

    fig, axes = plt.subplots(2, 5, figsize=(7, 3.2))

    datasets = [('Amazon-Books', books), ('Amazon-Movies', movies)]
    models = model_names_display

    for row_idx, (dataset_name, df) in enumerate(datasets):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            ndcg_col = get_ndcg_col(df, model)

            if ndcg_col is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                if row_idx == 0:
                    ax.set_title(model, fontsize=9, fontweight='bold')
                continue

            x = df['NO@K'].values
            y = df[ndcg_col].values

            # Drop NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            # Scatter
            ax.scatter(x, y, s=14, alpha=0.65, color='#2E86C1', edgecolors='white',
                      linewidth=0.3, zorder=3)

            # Regression line
            if len(x) > 2:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, color='#E74C3C',
                       linewidth=1.0, alpha=0.8, zorder=2)

                # Spearman
                rho, pval = stats.spearmanr(x, y)

                # Place annotation in the least-crowded corner
                # Use the actual axis limits (with padding) and check
                # only the 30% strip near each corner, not the full quadrant
                x_lo, x_hi = ax.get_xlim()
                y_lo_ax, y_hi_ax = ax.get_ylim()
                x_thresh = 0.30 * (x_hi - x_lo)
                y_thresh = 0.30 * (y_hi_ax - y_lo_ax)
                n_top_left = np.sum((x < x_lo + x_thresh) & (y > y_hi_ax - y_thresh))
                n_top_right = np.sum((x > x_hi - x_thresh) & (y > y_hi_ax - y_thresh))
                n_bot_left = np.sum((x < x_lo + x_thresh) & (y < y_lo_ax + y_thresh))
                n_bot_right = np.sum((x > x_hi - x_thresh) & (y < y_lo_ax + y_thresh))
                corners = {
                    'top_left': (0.05, 0.95, 'top', 'left', n_top_left),
                    'top_right': (0.95, 0.95, 'top', 'right', n_top_right),
                    'bot_left': (0.05, 0.05, 'bottom', 'left', n_bot_left),
                    'bot_right': (0.95, 0.05, 'bottom', 'right', n_bot_right),
                }
                # Manual override for specific subplots
                overrides = {(0, 2): 'top_right'}  # Amazon-Books + KAR
                if (row_idx, col_idx) in overrides:
                    best = corners[overrides[(row_idx, col_idx)]]
                else:
                    best = min(corners.values(), key=lambda c: c[4])
                ax.text(best[0], best[1], f'$\\rho$={rho:.2f}',
                       transform=ax.transAxes, fontsize=7,
                       verticalalignment=best[2], horizontalalignment=best[3],
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                alpha=0.6, edgecolor='#cccccc', linewidth=0.5))

            if row_idx == 0:
                ax.set_title(model, fontsize=9, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel('NDCG@10', fontsize=8)
            else:
                ax.set_ylabel('')

            if row_idx == 1:
                ax.set_xlabel('NO@K', fontsize=8)
            else:
                ax.set_xlabel('')

            ax.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=6.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            # Add y-axis buffer so the annotation doesn't cover points
            y_lo, y_hi = ax.get_ylim()
            y_pad = (y_hi - y_lo) * 0.20
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
            # Re-run locator after expanding limits so ticks cover the new range
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

    # Row labels
    fig.text(0.005, 0.72, 'Amazon-Books', rotation=90, fontsize=9, fontweight='bold',
             va='center', ha='center')
    fig.text(0.005, 0.28, 'Amazon-Movies', rotation=90, fontsize=9, fontweight='bold',
             va='center', ha='center')

    fig.tight_layout(rect=[0.02, 0, 1, 1])
    fig.subplots_adjust(hspace=0.35, wspace=0.45)
    fig.savefig(f'{FIGURES_DIR}/scatter_nok_ndcg.pdf', format='pdf')
    plt.close(fig)
    print("Figure 2 saved: scatter_nok_ndcg.pdf")


# ============================================================================
# Figure 3: Panel Size Sensitivity
# ============================================================================
def make_figure3():
    books = pd.read_csv(f'{RESULTS_DIR}/amazon-books_panel_ablation.csv')
    movies = pd.read_csv(f'{RESULTS_DIR}/amazon-movies_panel_ablation.csv')

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    # Model display names and colors
    model_map = {
        'AutoInt': 'AutoInt',
        'DCNv2': 'DCNv2',
        'KAR': 'KAR',
        'LLM-Rec': 'LLM-Rec',
        'UniSRec_transductive_ft': 'UniSRec',
    }
    colors = {
        'AutoInt': '#E74C3C',
        'DCNv2': '#3498DB',
        'KAR': '#2ECC71',
        'LLM-Rec': '#9B59B6',
        'UniSRec_transductive_ft': '#F39C12',
    }
    markers = {
        'AutoInt': 'o',
        'DCNv2': 's',
        'KAR': '^',
        'LLM-Rec': 'D',
        'UniSRec_transductive_ft': 'v',
    }

    datasets = [('Amazon-Books', books), ('Amazon-Movies', movies)]

    for idx, (name, df) in enumerate(datasets):
        ax = axes[idx]
        for model_key in model_map:
            mdf = df[df['model'] == model_key].sort_values('panel_count')
            if mdf.empty:
                continue

            x = mdf['panel_count'].values
            y = mdf['mean_rho'].values
            yerr = mdf['std_rho'].values

            ax.errorbar(x, y, yerr=yerr,
                       label=model_map[model_key],
                       color=colors[model_key],
                       marker=markers[model_key],
                       markersize=4.5,
                       linewidth=1.2,
                       capsize=2.5,
                       capthick=0.8,
                       elinewidth=0.8,
                       alpha=0.85)

        ax.set_xlabel('Panel Size', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Spearman $\\rho$', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks([50, 100, 200, 500, 1000, 2000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.5)

        if idx == 1:
            ax.legend(loc='lower right', fontsize=7, framealpha=0.9,
                     edgecolor='#cccccc', ncol=1)

    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/panel_sensitivity.pdf', format='pdf')
    plt.close(fig)
    print("Figure 3 saved: panel_sensitivity.pdf")


# ============================================================================
# Figure 4: Concordance Heatmap
# ============================================================================
def make_figure4():
    books = pd.read_csv(f'{RESULTS_DIR}/amazon-books_concordance.csv')
    movies = pd.read_csv(f'{RESULTS_DIR}/amazon-movies_concordance.csv')
    combined = pd.concat([books, movies], ignore_index=True)

    # Proxy metrics in desired order
    proxy_order = [
        'NO@K', 'RNO@K', 'DCG_comp', 'R2_incremental',
        'avg_tag_count', 'tag_diversity', 'avg_tag_length',
        'semantic_diversity', 'inter_item_similarity', 'tag_interaction_alignment'
    ]

    # Proxy display names (cleaner)
    proxy_display = {
        'NO@K': 'NO@K',
        'RNO@K': 'RNO@K',
        'DCG_comp': r'DCG$_{\mathrm{comp}}$',
        'R2_incremental': r'$R^2_{\mathrm{incr}}$',
        'avg_tag_count': 'Avg Tag Count',
        'tag_diversity': 'Tag Diversity',
        'avg_tag_length': 'Avg Tag Length',
        'semantic_diversity': 'Semantic Div.',
        'inter_item_similarity': 'Inter-Item Sim.',
        'tag_interaction_alignment': 'Tag-Interact. Align.',
    }

    # Model names mapping
    model_map = {
        'AutoInt': 'AutoInt',
        'DCNv2': 'DCNv2',
        'KAR': 'KAR',
        'LLM-Rec': 'LLM-Rec',
        'UniSRec_transductive_ft': 'UniSRec',
    }

    model_order = ['AutoInt', 'DCNv2', 'KAR', 'LLM-Rec', 'UniSRec_transductive_ft']
    dataset_order = ['amazon-books', 'amazon-movies']
    dataset_display = {'amazon-books': 'Books', 'amazon-movies': 'Movies'}

    # Build the heatmap matrix
    columns = []
    for ds in dataset_order:
        for m in model_order:
            columns.append(f"{model_map[m]}-{dataset_display[ds]}")

    n_rows = len(proxy_order)
    n_cols = len(columns)
    conc_matrix = np.full((n_rows, n_cols), np.nan)
    sig_matrix = np.full((n_rows, n_cols), '', dtype=object)

    for i, proxy in enumerate(proxy_order):
        for j, (ds, m) in enumerate([(ds, m) for ds in dataset_order for m in model_order]):
            mask = (combined['dataset'] == ds) & (combined['model'] == m) & (combined['proxy'] == proxy)
            rows = combined[mask]
            if len(rows) == 0:
                continue
            row = rows.iloc[0]
            conc_matrix[i, j] = row['concordance']
            pval = row['p_value_binomial']
            if pval < 0.01:
                sig_matrix[i, j] = '**'
            elif pval < 0.05:
                sig_matrix[i, j] = '*'

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.8))

    # Diverging colormap centered at 0.5: red (low/bad) → white (0.5) → blue (high/good)
    cmap = sns.diverging_palette(10, 240, as_cmap=True)

    im = ax.imshow(conc_matrix, cmap=cmap, aspect='auto',
                   vmin=0.3, vmax=0.9)

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = conc_matrix[i, j]
            if np.isnan(val):
                continue
            sig = sig_matrix[i, j]
            # Choose text color based on cell value
            text_color = 'white' if val > 0.78 or val < 0.38 else 'black'
            text = f'{val:.2f}'
            if sig:
                text += f'\n{sig}'
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=6, color=text_color, fontweight='normal')

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(columns, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([proxy_display[p] for p in proxy_order], fontsize=7.5)

    # Add vertical separator between datasets
    ax.axvline(x=4.5, color='white', linewidth=2)

    # Dataset group labels
    ax.text(2, -1.2, 'Amazon-Books', ha='center', va='center', fontsize=9,
            fontweight='bold')
    ax.text(7, -1.2, 'Amazon-Movies', ha='center', va='center', fontsize=9,
            fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Concordance', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Add a horizontal line at 0.5 in the colorbar (random baseline)
    cbar.ax.axhline(y=0.5, color='black', linewidth=0.8, linestyle='--')

    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/concordance_heatmap.pdf', format='pdf')
    plt.close(fig)
    print("Figure 4 saved: concordance_heatmap.pdf")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    print("\nAll figures generated successfully.")
