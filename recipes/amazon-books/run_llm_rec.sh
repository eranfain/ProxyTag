#!/bin/bash
# ==============================================================================
# LLM-Rec Baseline: Amazon Books
#
# Uses LLM-generated tags with mean pooling (no user-conditioned attention)
# and no item ID embeddings. This is a tag-only baseline.
# Paper: d=64, lr=1e-3, max_tags=36
# ==============================================================================
set -e

# --- Configuration ---
DATASET="amazon-books"
DATASET_DIR="data/${DATASET}"
USER_ID_COL="userId"
ITEM_ID_COL="asin"
TAGS_FILE="${DATASET_DIR}/tags_llm_rec_item_tags.parquet"

# --- Hyperparameters ---
HIDDEN_DIM=64
N_HEADS=4
MAX_TAGS=36
BATCH_SIZE=1024
LR=1e-3
MAX_EPOCHS=20
N_TRAIN_NEG=4
N_TEST_NEG=999
K=10

# --- Run ---
python scripts/main.py \
  --job_name "${DATASET}-llm-rec" \
  --train_path ${DATASET_DIR}/interactions/train.parquet \
  --val_path ${DATASET_DIR}/interactions/val.parquet \
  --test_path ${DATASET_DIR}/interactions/test.parquet \
  --items_data_path ${TAGS_FILE} \
  --user_id_col ${USER_ID_COL} \
  --item_id_col ${ITEM_ID_COL} \
  --sentence_transformers_model all-MiniLM-L6-v2 \
  --max_tags ${MAX_TAGS} \
  --hidden_dim ${HIDDEN_DIM} \
  --n_heads ${N_HEADS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --negative_sampler_n_items_train ${N_TRAIN_NEG} \
  --negative_sampler_n_items_test ${N_TEST_NEG} \
  --K ${K} \
  --max_epochs ${MAX_EPOCHS} \
  --no_use_item_id \
  --tag_encoding_type mean_pooling
