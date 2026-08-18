#!/usr/bin/env bash
set -euo pipefail
set -x

# ============================================================
# Qwen3-4B Math GRPO Teacher
#
# Goal:
#   Qwen3-4B-Non-Thinking
#       -> GRPO on DeepMath filtered level >= 6
#       -> Qwen3-4B-RL-Math teacher
#
# Hardware:
#   4 x H100 80GB
#
# Based on:
#   RUCBM/G-OPD
#   verl/examples/grpo_trainer/run_qwen3-4b-math.sh
# ============================================================


# ------------------------------------------------------------
# 0. Project root
# ------------------------------------------------------------
# Assume this script is placed under:
#   ./bash/run_qwen3_4b_math_grpo.sh
#
# Then PROJECT_ROOT is the repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false


# ------------------------------------------------------------
# 1. GPU
# ------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,2

NUM_GPUS=2


# ------------------------------------------------------------
# 2. SwanLab
# ------------------------------------------------------------
# Do not hard-code API key here.
# Run `swanlab login` beforehand, or:
#   export SWANLAB_API_KEY="..."
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${PROJECT_ROOT}/swanlog}"


# ------------------------------------------------------------
# 3. Paths
# ------------------------------------------------------------
DATA_ROOT="${PROJECT_ROOT}/data"

TRAIN_FILE="${DATA_ROOT}/DeepMath-103K/train_filtered_level6_datasets3.parquet"

AIME24_FILE="${DATA_ROOT}/AIME2024/test_datasets3.parquet"
AIME25_FILE="${DATA_ROOT}/AIME2025/test_datasets3.parquet"

VAL_FILES="['${AIME24_FILE}', '${AIME25_FILE}']"


# ------------------------------------------------------------
# 4. Model
# ------------------------------------------------------------
# You can override this with a local model path:
#
#   MODEL_PATH=/path/to/Qwen3-4B \
#       bash ./bash/run_qwen3_4b_math_grpo.sh
#
MODEL_PATH="${PROJECT_ROOT}/model/Qwen3-4B"


# ------------------------------------------------------------
# 5. Experiment
# ------------------------------------------------------------
PROJECT_NAME="g-opd-reproduction"
EXPERIMENT_NAME="qwen3_4b_non_thinking_deepmath_grpo_bsz128_n8_lr1e-6"

CKPT_DIR="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"


# ------------------------------------------------------------
# 6. Sanity check
# ------------------------------------------------------------
echo "============================================================"
echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "MODEL_PATH   : ${MODEL_PATH}"
echo "TRAIN_FILE   : ${TRAIN_FILE}"
echo "AIME24       : ${AIME24_FILE}"
echo "AIME25       : ${AIME25_FILE}"
echo "CKPT_DIR     : ${CKPT_DIR}"
echo "GPUs         : ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

test -f "${TRAIN_FILE}" || {
    echo "ERROR: training file not found: ${TRAIN_FILE}"
    exit 1
}

test -f "${AIME24_FILE}" || {
    echo "ERROR: AIME24 file not found: ${AIME24_FILE}"
    exit 1
}

test -f "${AIME25_FILE}" || {
    echo "ERROR: AIME25 file not found: ${AIME25_FILE}"
    exit 1
}


# ------------------------------------------------------------
# 7. GRPO training
# ------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=18432 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    reward_model.reward_manager=naive \
    \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations=10 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=50 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.total_training_steps=500 \
    "$@"
