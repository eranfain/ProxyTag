#!/bin/bash
# ==============================================================================
# AutoInt Baseline: Amazon Books
#
# Self-attention over user/item embeddings for feature interaction learning.
# Song et al., "AutoInt: Automatic Feature Interaction Learning via
# Self-Attentive Neural Networks"
# Paper: d=64
# ==============================================================================
set -e

# --- Configuration ---
DATASET="amazon-books"
DATASET_DIR="data/${DATASET}"
USER_ID_COL="userId"
ITEM_ID_COL="asin"
RESULTS_DIR="results/${DATASET}"

# --- Hyperparameters ---
EMBED_DIM=64
N_HEADS=8
N_LAYERS=3
BATCH_SIZE=256
MAX_EPOCHS=20
N_TEST_NEG=999
K=10

# --- Run ---
python -m proxytag.baselines.autoint \
  --job_name "${DATASET}-autoint" \
  --train_path ${DATASET_DIR}/interactions/train.parquet \
  --val_path ${DATASET_DIR}/interactions/val.parquet \
  --test_path ${DATASET_DIR}/interactions/test.parquet \
  --user_id_col ${USER_ID_COL} \
  --item_id_col ${ITEM_ID_COL} \
  --embed_dim ${EMBED_DIM} \
  --n_heads ${N_HEADS} \
  --n_layers ${N_LAYERS} \
  --batch_size ${BATCH_SIZE} \
  --max_epochs ${MAX_EPOCHS} \
  --n_test_neg ${N_TEST_NEG} \
  --K ${K} \
  --ckpt_dir checkpoints/${DATASET}/autoint \
  --results_dir ${RESULTS_DIR}
