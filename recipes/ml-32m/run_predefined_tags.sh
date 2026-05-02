#!/bin/bash
# ==============================================================================
# Predefined Tags Baseline: MovieLens-32M
#
# Uses predefined/existing tags (e.g., genres) instead of LLM-generated tags.
# Tests whether LLM tags provide value over readily available metadata.
# Paper: d=64, lr=1e-4, max_tags=24
# ==============================================================================
set -e

# --- Configuration ---
DATASET="ml-32m"
DATASET_DIR="data/${DATASET}"
USER_ID_COL="userId"
ITEM_ID_COL="movieId"
TAGS_FILE="${DATASET_DIR}/tags_genres.parquet"

# --- Hyperparameters ---
HIDDEN_DIM=64
MAX_TAGS=24
BATCH_SIZE=1024
LR=1e-4
MAX_EPOCHS=20
N_TRAIN_NEG=20
N_TEST_NEG=999
K=10

# --- Run ---
python scripts/main.py \
  --job_name "${DATASET}-predefined-tags" \
  --train_path ${DATASET_DIR}/interactions/train.parquet \
  --val_path ${DATASET_DIR}/interactions/val.parquet \
  --test_path ${DATASET_DIR}/interactions/test.parquet \
  --items_data_path ${TAGS_FILE} \
  --user_id_col ${USER_ID_COL} \
  --item_id_col ${ITEM_ID_COL} \
  --hidden_dim ${HIDDEN_DIM} \
  --max_tags ${MAX_TAGS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --negative_sampler_n_items_train ${N_TRAIN_NEG} \
  --negative_sampler_n_items_test ${N_TEST_NEG} \
  --K ${K} \
  --max_epochs ${MAX_EPOCHS} \
  --sentence_transformers_model all-MiniLM-L6-v2
