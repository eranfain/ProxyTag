#!/bin/bash
# ==============================================================================
# DCN-v2 Baseline
#
# Deep & Cross Network v2 with matrix-parameterized cross layers.
# Wang et al., "DCN V2: Improved Deep & Cross Network"
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
N_CROSS_LAYERS=3
STRUCTURE="parallel"
BATCH_SIZE=256
MAX_EPOCHS=20
N_TEST_NEG=1000
K=10

# --- Run ---
python -m proxytag.baselines.dcnv2 \
  --job_name "${DATASET}-dcnv2" \
  --train_path ${DATASET_DIR}/train.parquet \
  --val_path ${DATASET_DIR}/val.parquet \
  --test_path ${DATASET_DIR}/test.parquet \
  --user_id_col ${USER_ID_COL} \
  --item_id_col ${ITEM_ID_COL} \
  --embed_dim ${EMBED_DIM} \
  --n_cross_layers ${N_CROSS_LAYERS} \
  --structure ${STRUCTURE} \
  --batch_size ${BATCH_SIZE} \
  --max_epochs ${MAX_EPOCHS} \
  --n_test_neg ${N_TEST_NEG} \
  --K ${K} \
  --ckpt_dir checkpoints/${DATASET}/dcnv2 \
  --results_dir ${RESULTS_DIR}
