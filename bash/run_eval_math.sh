#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Evaluate Qwen3-4B Non-Thinking RL-Math teacher
#
# Benchmarks:
#   - AIME 2024
#   - AIME 2025
#
# Evaluation protocol follows G-OPD:
#   temperature = 1.0
#   top_p       = 1.0
#   n           = 32
#   max_tokens  = 16384
#   seed        = 42
#
# GPUs:
#   AIME24 -> GPU 0
#   AIME25 -> GPU 2
# ============================================================

# -------------------------
# 0. Project root
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# -------------------------
# 1. Model
# -------------------------
MODEL_PATH="${PROJECT_ROOT}/model/Qwen3-4B-Non-Thinking-RL-Math"
MODEL_NAME="Qwen3-4B-Non-Thinking-RL-Math-Step500"

# -------------------------
# 2. Evaluator
# -------------------------
EVAL_PY="${PROJECT_ROOT}/math_eval/eval_math.py"

# -------------------------
# 3. Data
# -------------------------
AIME24_FILE="${PROJECT_ROOT}/data/AIME2024/test.jsonl"
AIME25_FILE="${PROJECT_ROOT}/data/AIME2025/test.jsonl"

# -------------------------
# 4. Output
# -------------------------
OUTPUT_ROOT="${PROJECT_ROOT}/eval_outputs/${MODEL_NAME}"

AIME24_OUTPUT="${OUTPUT_ROOT}/aime24.jsonl"
AIME25_OUTPUT="${OUTPUT_ROOT}/aime25.jsonl"

mkdir -p "${OUTPUT_ROOT}"

# -------------------------
# 5. Sanity checks
# -------------------------
echo "============================================================"
echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "MODEL_PATH   : ${MODEL_PATH}"
echo "MODEL_NAME   : ${MODEL_NAME}"
echo "AIME24       : ${AIME24_FILE}"
echo "AIME25       : ${AIME25_FILE}"
echo "OUTPUT_ROOT  : ${OUTPUT_ROOT}"
echo "GPUs         : 0,2"
echo "============================================================"

test -d "${MODEL_PATH}" || {
    echo "[ERROR] Model directory not found:"
    echo "        ${MODEL_PATH}"
    exit 1
}

test -f "${EVAL_PY}" || {
    echo "[ERROR] eval_math.py not found:"
    echo "        ${EVAL_PY}"
    exit 1
}

test -f "${AIME24_FILE}" || {
    echo "[ERROR] AIME24 jsonl not found:"
    echo "        ${AIME24_FILE}"
    exit 1
}

test -f "${AIME25_FILE}" || {
    echo "[ERROR] AIME25 jsonl not found:"
    echo "        ${AIME25_FILE}"
    exit 1
}

# ============================================================
# 6. AIME 2024
# GPU 0
# ============================================================

echo "[INFO] Starting AIME24 evaluation on GPU 0..."

CUDA_VISIBLE_DEVICES=0 python3 "${EVAL_PY}" \
    --input_file "${AIME24_FILE}" \
    --model_path "${MODEL_PATH}" \
    --output_file "${AIME24_OUTPUT}" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 \
    --seed 42 \
    > "${OUTPUT_ROOT}/aime24.log" 2>&1 &

PID_AIME24=$!

echo "[INFO] AIME24 PID: ${PID_AIME24}"


# ============================================================
# 7. AIME 2025
# GPU 2
# ============================================================

echo "[INFO] Starting AIME25 evaluation on GPU 2..."

CUDA_VISIBLE_DEVICES=2 python3 "${EVAL_PY}" \
    --input_file "${AIME25_FILE}" \
    --model_path "${MODEL_PATH}" \
    --output_file "${AIME25_OUTPUT}" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 \
    --seed 42 \
    > "${OUTPUT_ROOT}/aime25.log" 2>&1 &

PID_AIME25=$!

echo "[INFO] AIME25 PID: ${PID_AIME25}"


# ============================================================
# 8. Wait for both evaluations
# ============================================================

echo "============================================================"
echo "[INFO] Both evaluations are running."
echo "[INFO] Waiting for completion..."
echo "============================================================"

wait "${PID_AIME24}"
STATUS_AIME24=$?

wait "${PID_AIME25}"
STATUS_AIME25=$?

echo
echo "============================================================"
echo "Evaluation finished"
echo "------------------------------------------------------------"
echo "AIME24 exit code : ${STATUS_AIME24}"
echo "AIME25 exit code : ${STATUS_AIME25}"
echo
echo "AIME24 log:"
echo "  ${OUTPUT_ROOT}/aime24.log"
echo
echo "AIME25 log:"
echo "  ${OUTPUT_ROOT}/aime25.log"
echo
echo "Outputs:"
echo "  ${AIME24_OUTPUT}"
echo "  ${AIME25_OUTPUT}"
echo "============================================================"
