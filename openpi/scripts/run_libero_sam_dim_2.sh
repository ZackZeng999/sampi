#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline:
#   extract_libero_sam_prompts.py -> build_libero_sam_mask_cache.py -> render_libero_sam_dim_dataset.py
#
# Before running, edit BLACKLIST_JSON if needed:
#   /root/autodl-tmp/datasets/physical-intelligence/libero_sam_masks_2/blacklist.json

PYTHON_BIN="${PYTHON_BIN:-/root/proj/openpi/.venv/bin/python}"
OPENPI_ROOT="${OPENPI_ROOT:-/root/proj/openpi}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/datasets/physical-intelligence/libero}"

# Step 1 output: prompt sidecars. Kept as requested, even though the name says masks.
PROMPTS_ROOT="${PROMPTS_ROOT:-/root/autodl-tmp/datasets/physical-intelligence/libero_sam_masks_2}"
BLACKLIST_JSON="${BLACKLIST_JSON:-${PROMPTS_ROOT}/blacklist.json}"

# Step 2 output: actual per-episode SAM mask cache.
MASK_CACHE_ROOT="${MASK_CACHE_ROOT:-/root/autodl-tmp/datasets/physical-intelligence/libero_sam_mask_cache_2}"

# Step 3 output: rendered LeRobot dataset with SAM-dimmed base images.
DIM_DATASET_ROOT="${DIM_DATASET_ROOT:-/root/autodl-tmp/datasets/physical-intelligence/libero_sam_dim_2}"
DIM_REPO_ID="${DIM_REPO_ID:-physical-intelligence/libero_sam_dim_2}"

EXTRACT_URL="${EXTRACT_URL:-http://127.0.0.1:9001/extract}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/root/autodl-tmp/sam3_model/sam3.pt}"

MAX_PROMPTS="${MAX_PROMPTS:-5}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
PROMPT_GROUP_SIZE="${PROMPT_GROUP_SIZE:-10}"
PROMPT_GROUP_SEED="${PROMPT_GROUP_SEED:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"

SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.6}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.0}"
MAX_MASKS_PER_PROMPT="${MAX_MASKS_PER_PROMPT:-3}"
FRAME_BATCH_SIZE="${FRAME_BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"

BACKGROUND_SCALE="${BACKGROUND_SCALE:-0.4}"
BLUR_RADIUS="${BLUR_RADIUS:-1.5}"

cd "${OPENPI_ROOT}"

mkdir -p "${PROMPTS_ROOT}"
if [[ ! -f "${BLACKLIST_JSON}" ]]; then
  printf '[\n]\n' > "${BLACKLIST_JSON}"
fi

# echo "[1/3] Extracting SAM prompts into ${PROMPTS_ROOT}"
# "${PYTHON_BIN}" scripts/extract_libero_sam_prompts.py \
#   --dataset-root "${DATASET_ROOT}" \
#   --output-root "${PROMPTS_ROOT}" \
#   --extract-url "${EXTRACT_URL}" \
#   --max-prompts "${MAX_PROMPTS}" \
#   --max-rounds "${MAX_ROUNDS}" \
#   --timeout-sec "${TIMEOUT_SEC}" \
#   --prompt-group-size "${PROMPT_GROUP_SIZE}" \
#   --prompt-group-seed "${PROMPT_GROUP_SEED}" \
#   --blacklist-path "${BLACKLIST_JSON}" \
#   --overwrite

# echo "[2/3] Building SAM mask cache into ${MASK_CACHE_ROOT}"
# "${PYTHON_BIN}" scripts/build_libero_sam_mask_cache.py \
#   --dataset-root "${DATASET_ROOT}" \
#   --prompts-root "${PROMPTS_ROOT}" \
#   --output-root "${MASK_CACHE_ROOT}" \
#   --checkpoint-path "${CHECKPOINT_PATH}" \
#   --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
#   --score-threshold "${SCORE_THRESHOLD}" \
#   --max-masks-per-prompt "${MAX_MASKS_PER_PROMPT}" \
#   --frame-batch-size "${FRAME_BATCH_SIZE}" \
#   --num-workers "${NUM_WORKERS}" \
#   --overwrite

echo "[3/3] Rendering SAM-dim dataset into ${DIM_DATASET_ROOT}"
"${PYTHON_BIN}" scripts/render_libero_sam_dim_dataset.py \
  --dataset-root "${DATASET_ROOT}" \
  --masks-root "${MASK_CACHE_ROOT}" \
  --output-root "${DIM_DATASET_ROOT}" \
  --repo-id "${DIM_REPO_ID}" \
  --background-scale "${BACKGROUND_SCALE}" \
  --blur-radius "${BLUR_RADIUS}" \
  --overwrite

echo "Done. Rendered dataset: ${DIM_DATASET_ROOT}"
