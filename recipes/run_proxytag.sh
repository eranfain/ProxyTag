#!/bin/bash
# ==============================================================================
# ProxyTag: Train and evaluate the full ProxyTag hybrid model
#
# Uses LLM-generated tags with user-conditioned attention (cross_encoder).
# This is the main model proposed in the paper.
# ==============================================================================
set -e

# --- Configuration (edit these) ---
DATASET="ml-32m"
DATASET_DIR="data/${DATASET}"
USER_ID_COL="userId"
ITEM_ID_COL="movieId"
TAGS_FILE="${DATASET_DIR}/tags_multi_use_knowledge_recommend_content.parquet"

# --- Hyperparameters ---
HIDDEN_DIM=256
N_HEADS=8
MAX_TAGS=36
BATCH_SIZE=256
LR=1e-3
MAX_EPOCHS=50
N_TRAIN_NEG=200
N_TEST_NEG=1000
K=10
EMBEDDING_REG=1e-5

# --- Run ---
python scripts/main.py \
  --job_name "${DATASET}-proxytag" \
  --train_path ${DATASET_DIR}/train.parquet \
  --val_path ${DATASET_DIR}/val.parquet \
  --test_path ${DATASET_DIR}/test.parquet \
  --items_data_path ${TAGS_FILE} \
  --user_id_col ${USER_ID_COL} \
  --item_id_col ${ITEM_ID_COL} \
  --sentence_transformers_model all-MiniLM-L6-v2 \
  --max_tags ${MAX_TAGS} \
  --hidden_dim ${HIDDEN_DIM} \
  --n_heads ${N_HEADS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --embedding_reg ${EMBEDDING_REG} \
  --negative_sampler_n_items_train ${N_TRAIN_NEG} \
  --negative_sampler_n_items_test ${N_TEST_NEG} \
  --K ${K} \
  --max_epochs ${MAX_EPOCHS}
