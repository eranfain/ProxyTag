# ProxyTag

Hybrid recommendation system that combines collaborative filtering with LLM-generated tag-based content encoding. Uses a proxy metric to efficiently evaluate tag quality without expensive model training.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Pipeline Overview

The full pipeline runs in this order:

1. **Data preparation** - Split interactions into train/val/test
2. **Tag generation** - Generate tags using an LLM
3. **Proxy metric** - Evaluate tag quality (fast, no training)
4. **Model training** - Train and evaluate the hybrid model
5. **Baselines** - Train and evaluate baseline models
6. **Analysis** - Statistical tests, bucket analysis

---

## 1. Data Preparation

Split raw interaction data into temporal train/val/test sets (80/10/10):

```bash
python scripts/prepare_data.py \
  --interactions_path raw_data/ratings.csv \
  --output_dir data/ml-32m \
  --user_id_col userId \
  --item_id_col movieId \
  --create_tiny
```

**Amazon Books:**
```bash
python scripts/prepare_data.py \
  --interactions_path raw_data/ratings.csv \
  --output_dir data/amazon-books \
  --user_id_col userId \
  --item_id_col asin \
  --create_tiny
```

## 2. Tag Generation

Generate tags using Azure OpenAI. Requires `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` environment variables.

```bash
export AZURE_OPENAI_API_KEY=your_key
export AZURE_OPENAI_ENDPOINT=your_endpoint

# MovieLens
python scripts/generate_tags.py \
  --dataset movielens \
  --input_path raw_data/movies.parquet \
  --output_dir data/ml-32m/tags/full \
  --item_id_col movieId \
  --title_col title \
  --description_col overview

# Amazon Books
python scripts/generate_tags.py \
  --dataset amazon-books \
  --input_path raw_data/books_metadata.parquet \
  --output_dir data/amazon-books/tags/full \
  --item_id_col asin \
  --title_col title \
  --description_col description
```

Use `--list_prompts` to see available prompt variants. Use `--prompts` to run specific ones.

## 3. Proxy Metric

Evaluate tag quality by measuring alignment between tag-based and CF-based item similarities. This is a fast screening step (minutes) that predicts downstream hybrid performance without training.

```bash
# First, train a CF-only model (needed for CF embeddings)
python scripts/main.py \
  --job_name cf-baseline \
  --train_path data/ml-32m/interactions/train.parquet \
  --val_path data/ml-32m/interactions/val.parquet \
  --test_path data/ml-32m/interactions/test.parquet \
  --user_id_col userId --item_id_col movieId \
  --no_use_tags \
  --sentence_transformers_model all-MiniLM-L6-v2 \
  --max_epochs 20

# Run proxy metric
python -m proxytag.analysis.proxy_metric \
  --train_path data/ml-32m/interactions/train.parquet \
  --items_data_path data/ml-32m/tags_multi.parquet \
  --user_id_col userId \
  --item_id_col movieId \
  --panel_size 0.05 \
  --use_interaction_cf \
  --K 10
```

**Interpreting results:**
- **NO@K** (Neighbour Overlap at K): Fraction of top-K neighbours shared between CF-based and tag-based rankings
- Ideal tags: moderate NO@K (0.3-0.5) -- aligned but complementary to CF

## 4. Training the ProxyTag Model

The full hybrid model combines user/item embeddings with user-conditioned attention over tag embeddings:

```bash
python scripts/main.py \
  --job_name ml-32m-proxytag \
  --train_path data/ml-32m/interactions/train.parquet \
  --val_path data/ml-32m/interactions/val.parquet \
  --test_path data/ml-32m/interactions/test.parquet \
  --items_data_path data/ml-32m/tags_multi_use_knowledge_recommend_mood_style.parquet \
  --user_id_col userId \
  --item_id_col movieId \
  --sentence_transformers_model all-MiniLM-L6-v2 \
  --max_tags 24 \
  --hidden_dim 64 \
  --n_heads 4 \
  --batch_size 256 \
  --lr 1e-4 \
  --embedding_reg 1e-5 \
  --negative_sampler_n_items_train 200 \
  --negative_sampler_n_items_test 999 \
  --K 10 \
  --max_epochs 50
```

Evaluation runs automatically after training, reporting Precision@K, Recall@K, and NDCG@K with popularity bucket breakdown. Results are saved to `results/<job_name>_results.json`.

**Ablation flags:**
- `--no_use_item_id` -- tags-only mode (no item ID embeddings)
- `--no_use_tags` -- CF-only mode (no tag encoding)
- `--tag_encoding_type mean_pooling` -- simple mean pooling instead of cross-attention

See `recipes/run_proxytag.sh` for a ready-to-run script.

## 5. Baselines

### AutoInt

Self-attention over user/item embeddings for automatic feature interaction learning:

```bash
python -m proxytag.baselines.autoint \
  --job_name ml-32m-autoint \
  --train_path data/ml-32m/interactions/train.parquet \
  --val_path data/ml-32m/interactions/val.parquet \
  --test_path data/ml-32m/interactions/test.parquet \
  --user_id_col userId --item_id_col movieId \
  --embed_dim 64 --n_heads 8 --n_layers 3 \
  --batch_size 256 --max_epochs 20 \
  --n_test_neg 999 --K 10 \
  --ckpt_dir checkpoints/autoint \
  --results_dir results
```

### DCN-v2

Deep & Cross Network v2 with matrix-parameterized cross layers:

```bash
python -m proxytag.baselines.dcnv2 \
  --job_name ml-32m-dcnv2 \
  --train_path data/ml-32m/interactions/train.parquet \
  --val_path data/ml-32m/interactions/val.parquet \
  --test_path data/ml-32m/interactions/test.parquet \
  --user_id_col userId --item_id_col movieId \
  --embed_dim 64 --n_cross_layers 3 --structure parallel \
  --batch_size 256 --max_epochs 20 \
  --n_test_neg 999 --K 10 \
  --ckpt_dir checkpoints/dcnv2 \
  --results_dir results
```

### Other baselines (via main.py)

These baselines use the ProxyTag architecture with different configurations:

| Baseline | Key flags | Recipe |
|----------|-----------|--------|
| **LLM-Rec** | `--no_use_item_id --tag_encoding_type mean_pooling` | `recipes/run_llm_rec.sh` |
| **CF-only** | `--no_use_tags` | `recipes/run_cf_only.sh` |
| **Predefined tags** | Use `tags_categories.parquet` or `tags_genres.parquet` | `recipes/run_predefined_tags.sh` |

## 6. Analysis

### Aggregate Results

Collect all experiment results into a single CSV:

```bash
python -m proxytag.analysis.aggregate_results
```

### Significance Test

Paired t-test between two models on per-user metrics:

```bash
python -m proxytag.analysis.significance_test \
  --model_a_ckpt checkpoints/proxytag/best.ckpt --model_a_type hybrid \
  --model_a_name "ProxyTag" \
  --model_b_ckpt checkpoints/autoint/best.ckpt --model_b_type autoint \
  --model_b_name "AutoInt" \
  --train_path data/ml-32m/interactions/train.parquet \
  --val_path data/ml-32m/interactions/val.parquet \
  --test_path data/ml-32m/interactions/test.parquet \
  --items_data_path data/ml-32m/tags_multi.parquet \
  --user_id_col userId --item_id_col movieId \
  --max_users 5000 --K 10 --n_test_neg 999
```

Reports mean, std, t-statistic, p-value, and Cohen's d for Recall@K and NDCG@K.

### Bucket Analysis

Analyze how proxy metric NO@K varies across item popularity buckets:

```bash
python -m proxytag.analysis.analyze_bucket_correlations \
  --proxy_results_dir proxy_results \
  --hybrid_results_dir results
```

### Create Panel Files

Create panel item subsets for proxy metric evaluation:

```bash
python -m proxytag.analysis.create_panel_files \
  --dataset_dir data/ml-32m \
  --item_id_col movieId \
  --panel_size 0.05
```

## Recipe Scripts

All recipes are in the `recipes/` directory with configurable variables at the top:

| Script | Description |
|--------|-------------|
| `run_proxytag.sh` | Full ProxyTag hybrid model |
| `run_autoint.sh` | AutoInt baseline |
| `run_dcnv2.sh` | DCN-v2 baseline |
| `run_llm_rec.sh` | LLM-Rec baseline (tags-only, mean pooling) |
| `run_cf_only.sh` | CF-only baseline (no tags) |
| `run_predefined_tags.sh` | Predefined tags baseline |

Edit the variables at the top of each script to configure dataset paths and hyperparameters.

## Project Structure

```
ProxyTag/
├── proxytag/                   # Python package
│   ├── core/                   # Core modules
│   │   ├── data.py             # Data loading, ID mapping, negative sampling
│   │   ├── eval.py             # Evaluation metrics (Precision/Recall/NDCG@K)
│   │   ├── model.py            # HybridRecLightning model
│   │   ├── tags.py             # Tag embedding computation
│   │   └── utils.py            # Popularity percentiles
│   ├── baselines/              # Baseline models
│   │   ├── autoint.py          # AutoInt
│   │   └── dcnv2.py            # DCN-v2
│   └── analysis/               # Analysis scripts
│       ├── significance_test.py
│       ├── proxy_metric.py
│       ├── aggregate_results.py
│       └── ...
├── scripts/                    # Entry points
│   ├── main.py                 # Train/evaluate hybrid model
│   ├── prepare_data.py         # Data splitting
│   └── generate_tags.py        # LLM tag generation
├── recipes/                    # Ready-to-run shell scripts
├── pyproject.toml
└── requirements.txt
```
