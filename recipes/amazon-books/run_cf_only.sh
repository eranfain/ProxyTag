#!/bin/bash
# ==============================================================================
# CF-Only Baseline: Amazon Books
#
# Pure collaborative filtering with user/item ID embeddings only.
# No tag information is used. This is the --no_use_tags ablation.
# Paper: d=64, lr=1e-3
# ==============================================================================
set -e

# --- Configuration ---
DATASET="amazon-books"
DATASET_DIR="data/${DATASET}"
USER_ID_COL="userId"
ITEM_ID_COL="asin"

# --- Hyperparameters ---
HIDDEN_DIM=64
BATCH_SIZE=1024
LR=1e-3
MAX_EPOCHS=20
N_TRAIN_NEG=20
N_TEST_NEG=999
K=10

# --- Run ---
python scripts/main.py \
  --job_name "${DATASET}-cf-only" \
  --train_path ${DATASET_DIR}/interactions/train.parquet \
  --val_path ${DATASET_DIR}/interactions/val.parquet \
  --test_path ${DATASET_DIR}/interactions/test.parquet \
  --user_id_col ${USER_ID_COL} \
  --item_id_col ${ITEM_ID_COL} \
  --hidden_dim ${HIDDEN_DIM} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --negative_sampler_n_items_train ${N_TRAIN_NEG} \
  --negative_sampler_n_items_test ${N_TEST_NEG} \
  --K ${K} \
  --max_epochs ${MAX_EPOCHS} \
  --no_use_tags \
  --sentence_transformers_model all-MiniLM-L6-v2
