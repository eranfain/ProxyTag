#!/bin/bash
# ==============================================================================
# AutoInt Baseline
#
# Self-attention over user/item embeddings for feature interaction learning.
# Song et al., "AutoInt: Automatic Feature Interaction Learning via
# Self-Attentive Neural Networks"
# ==============================================================================
set -e

# --- Configuration (edit these) ---
DATASET="ml-32m"
DATASET_DIR="data/${DATASET}"
USER_ID_COL="userId"
ITEM_ID_COL="movieId"
RESULTS_DIR="results/${DATASET}"

# --- Hyperparameters ---
EMBED_DIM=64
N_HEADS=8
N_LAYERS=3
BATCH_SIZE=256
MAX_EPOCHS=20
N_TEST_NEG=1000
K=10

# --- Run ---
python -m proxytag.baselines.autoint \
  --job_name "${DATASET}-autoint" \
  --train_path ${DATASET_DIR}/train.parquet \
  --val_path ${DATASET_DIR}/val.parquet \
  --test_path ${DATASET_DIR}/test.parquet \
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
