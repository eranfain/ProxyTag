#!/bin/bash
# ==============================================================================
# Predefined Tags Baseline
#
# Uses predefined/existing tags (e.g., genres, categories) instead of
# LLM-generated tags. Tests whether LLM tags provide value over
# readily available metadata.
# ==============================================================================
set -e

# --- Configuration (edit these) ---
DATASET="amazon-books"
DATASET_DIR="data/${DATASET}"
USER_ID_COL="userId"
ITEM_ID_COL="asin"
TAGS_FILE="${DATASET_DIR}/tags_categories.parquet"

# --- Hyperparameters ---
HIDDEN_DIM=64
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
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --negative_sampler_n_items_train ${N_TRAIN_NEG} \
  --negative_sampler_n_items_test ${N_TEST_NEG} \
  --K ${K} \
  --max_epochs ${MAX_EPOCHS} \
  --sentence_transformers_model all-MiniLM-L6-v2
