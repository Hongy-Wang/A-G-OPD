#!/usr/bin/env bash
set -euo pipefail
set -x

# ============================================================
# G-OPD mechanism-analysis reproduction: Qwen3-4B -> 4B-RL-Math
#
# Paper setting:
#   Section 4.1: same-size single-teacher distillation
#
#   Student / pi_theta(0):
#       Qwen3-4B-Non-Thinking
#
#   Teacher / pi_T:
#       Qwen3-4B-Non-Thinking-RL-Math (Step500)
#
#   G-OPD reference / pi_ref:
#       Qwen3-4B-Non-Thinking (student initial checkpoint)
#
#   Train:
#       DeepMath-103K filtered difficulty >= 6
#
#   Paper hyperparameters:
#       batch size          = 1024
#       rollout n           = 1
#       max prompt length   = 2048
#       max response length = 16384
#       temperature         = 1.0
#       top-p               = 1.0
#       LR                  = 1e-5
#       optimization steps  = 50   (same-size setting)
#
# This A-G-OPD mechanism run additionally enables:
#   - all existing token-level G-OPD diagnostics
#   - OPD / extra / total gradient norms
#   - gradient cosine / parallel / orthogonal geometry
#   - diagnostics at EVERY optimizer step
#
# Examples:
#
#   # Standard OPD point
#   GOPD_LAM=1.0 bash ./bash/run_gopd_mechanism_qwen3_4b_math.sh
#
#   # Main ExOPD point
#   GOPD_LAM=1.25 bash ./bash/run_gopd_mechanism_qwen3_4b_math.sh
#
#   # Full lambda sweep used for mechanism analysis in the paper:
#   for lam in 0.25 0.5 0.75 1.0 1.25 1.5; do
#       GOPD_LAM="${lam}" \
#       bash ./bash/run_gopd_mechanism_qwen3_4b_math.sh
#   done
# ============================================================


# ============================================================
# 0. Environment
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${PROJECT_ROOT}/swanlog}"


# ============================================================
# 1. Models
# ============================================================

# Student = same 4B base model from which the RL teacher was obtained.
STUDENT_MODEL="${STUDENT_MODEL:-${PROJECT_ROOT}/model/Qwen3-4B}"

# Your downloaded public Step-500 RL math teacher.
TEACHER_MODEL="${TEACHER_MODEL:-${PROJECT_ROOT}/model/Qwen3-4B-Non-Thinking-RL-Math-Step500}"

# Same-size Section 4.1 uses the student's initial/base model as pi_ref.
GOPD_REF_MODEL="${GOPD_REF_MODEL:-${STUDENT_MODEL}}"


# ============================================================
# 2. Data
# ============================================================

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6_datasets3.parquet}"

AIME24_FILE="${AIME24_FILE:-${DATA_ROOT}/AIME2024/test_datasets3.parquet}"
AIME25_FILE="${AIME25_FILE:-${DATA_ROOT}/AIME2025/test_datasets3.parquet}"

# Match the released training script: AIME24 + AIME25 validation.
VAL_FILES="['${AIME24_FILE}', '${AIME25_FILE}']"


# ============================================================
# 3. G-OPD mechanism parameters
# ============================================================

# Main ExOPD setting in the paper.
GOPD_LAM="${GOPD_LAM:-1.25}"

# Same-size experiments in Section 4.1 use 50 optimizer steps.
TRAINING_STEPS="${TRAINING_STEPS:-50}"

# All gradient diagnostics ON, at every training step.
GOPD_GRAD_DIAG=true
GOPD_GRAD_DIAG_FREQ=1


# ============================================================
# 4. Paper training hyperparameters
# ============================================================

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1024}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"

ACTOR_LR="${ACTOR_LR:-1e-5}"

ROLLOUT_N=1
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_TOP_P=1.0

PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"

ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"

TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-50}"


# ============================================================
# 5. Experiment naming
# ============================================================

PROJECT_NAME="${PROJECT_NAME:-gopd-mechanism-analysis}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_to_4b_rl_math_gopd_lam${GOPD_LAM}_diag_all_step50_seed42}"

CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
VAL_OUTPUT_DIR="${VAL_OUTPUT_DIR:-${PROJECT_ROOT}/validation_outputs_gopd/${EXPERIMENT_NAME}}"

mkdir -p "${CKPT_DIR}"
mkdir -p "${VAL_OUTPUT_DIR}"


# ============================================================
# 6. Sanity checks
# ============================================================

echo "============================================================"
echo "G-OPD mechanism analysis: Qwen3-4B -> Qwen3-4B-RL-Math"
echo "------------------------------------------------------------"
echo "PROJECT_ROOT                  = ${PROJECT_ROOT}"
echo "STUDENT_MODEL                 = ${STUDENT_MODEL}"
echo "TEACHER_MODEL                 = ${TEACHER_MODEL}"
echo "GOPD_REF_MODEL                = ${GOPD_REF_MODEL}"
echo "TRAIN_FILE                    = ${TRAIN_FILE}"
echo "AIME24_FILE                   = ${AIME24_FILE}"
echo "AIME25_FILE                   = ${AIME25_FILE}"
echo "GOPD_LAM                      = ${GOPD_LAM}"
echo "TRAINING_STEPS                = ${TRAINING_STEPS}"
echo "TRAIN_BATCH_SIZE              = ${TRAIN_BATCH_SIZE}"
echo "PPO_MINI_BATCH_SIZE           = ${PPO_MINI_BATCH_SIZE}"
echo "PPO_MICRO_BATCH_SIZE_PER_GPU  = ${PPO_MICRO_BATCH_SIZE_PER_GPU}"
echo "MAX_PROMPT_LENGTH             = ${MAX_PROMPT_LENGTH}"
echo "MAX_RESPONSE_LENGTH           = ${MAX_RESPONSE_LENGTH}"
echo "ACTOR_LR                      = ${ACTOR_LR}"
echo "GOPD_GRAD_DIAG                = ${GOPD_GRAD_DIAG}"
echo "GOPD_GRAD_DIAG_FREQ           = ${GOPD_GRAD_DIAG_FREQ}"
echo "CUDA_VISIBLE_DEVICES          = ${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS                      = ${NUM_GPUS}"
echo "ROLLOUT_TP_SIZE               = ${ROLLOUT_TP_SIZE}"
echo "CKPT_DIR                      = ${CKPT_DIR}"
echo "============================================================"

for path in \
    "${TRAIN_FILE}" \
    "${AIME24_FILE}" \
    "${AIME25_FILE}"
do
    test -f "${path}" || {
        echo "ERROR: file not found: ${path}"
        exit 1
    }
done

for path in \
    "${STUDENT_MODEL}" \
    "${TEACHER_MODEL}" \
    "${GOPD_REF_MODEL}"
do
    test -d "${path}" || {
        echo "ERROR: model directory not found: ${path}"
        exit 1
    }
done


# ============================================================
# 7. Train
#
# A_GOPD
#   = (log pi_T - log pi_S)
#     + (lambda - 1) * (log pi_T - log pi_ref)
#
# Here:
#   pi_S     = current Qwen3-4B student
#   pi_T     = Qwen3-4B RL-Math teacher
#   pi_ref   = initial Qwen3-4B base
#
# lambda = 1.0:
#   standard OPD
#
# lambda > 1:
#   reward extrapolation (ExOPD)
# ============================================================

python3 -m verl.trainer.main_ppo \
    \
    algorithm.adv_estimator=gopd \
    algorithm.use_kl_in_reward=False \
    algorithm.gopd.lam="${GOPD_LAM}" \
    algorithm.gopd.grad_diagnostics.enable="${GOPD_GRAD_DIAG}" \
    algorithm.gopd.grad_diagnostics.freq="${GOPD_GRAD_DIAG_FREQ}" \
    \
    algorithm.privileged_opd.teacher.enable=false \
    algorithm.privileged_opd.gopd_ref.enable=false \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    +data.seed=42 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
    actor_rollout_ref.gopd_ref.model.path="${GOPD_REF_MODEL}" \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.top_k=-1 \
    \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    \
    reward_model.reward_manager=naive \
    \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="${VAL_OUTPUT_DIR}" \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations=10 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=3 \
    trainer.total_training_steps="${TRAINING_STEPS}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    "$@"
