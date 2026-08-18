#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Privileged RSOPD pilot training
#
#   Student : Qwen3-1.7B
#   Teacher : Qwen3-4B
#   Method  : Reverse sampled-token OPD
#             + privileged teacher information
#
#   Privileged information:
#       full_response
#
#   Train   : 1500 stratified MATH examples
#   Eval    : full MATH-500
#   GPUs    : 4 x H100
#
# NOTE:
# `full_response` is currently used as the first plumbing /
# oracle-style Privileged OPD baseline.
# ============================================================


# ============================================================
# 0. Environment
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false


# ============================================================
# SwanLab
# ============================================================

# Do NOT hard-code API keys here.
# Either run:
#
#   swanlab login
#
# beforehand, or export:
#
#   export SWANLAB_API_KEY="..."
#
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"


# ============================================================
# 1. Project paths
# ============================================================

VERL_ROOT="${VERL_ROOT:-$HOME/Code/LLM/verl-cu124}"

export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${VERL_ROOT}/swanlog}"

STUDENT_MODEL="${STUDENT_MODEL:-${VERL_ROOT}/model/Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-${VERL_ROOT}/model/Qwen3-4B}"

TRAIN_FILE="${TRAIN_FILE:-${VERL_ROOT}/data/MATH/math_train_1500_verl.parquet}"
VAL_FILE="${VAL_FILE:-${VERL_ROOT}/data/MATH-500/math500_verl.parquet}"


# ============================================================
# 2. Experiment naming
# ============================================================

PROJECT_NAME="${PROJECT_NAME:-privileged_rsopd_math}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1.7b_to_4b_math1500_full_response_seed42}"

CKPT_DIR="${CKPT_DIR:-${VERL_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"


cd "${VERL_ROOT}"


# ============================================================
# 3. Dependency checks
# ============================================================

python -c "import swanlab" >/dev/null 2>&1 || {
    echo "SwanLab is not installed in the current Python environment."
    echo 'Install it with:'
    echo 'uv pip install --python "$VIRTUAL_ENV/bin/python" swanlab'
    exit 1
}


# ============================================================
# 4. File sanity checks
# ============================================================

test -d "${STUDENT_MODEL}" || {
    echo "Student model not found:"
    echo "${STUDENT_MODEL}"
    exit 1
}

test -d "${TEACHER_MODEL}" || {
    echo "Teacher model not found:"
    echo "${TEACHER_MODEL}"
    exit 1
}

test -f "${TRAIN_FILE}" || {
    echo "Train parquet not found:"
    echo "${TRAIN_FILE}"
    exit 1
}

test -f "${VAL_FILE}" || {
    echo "Validation parquet not found:"
    echo "${VAL_FILE}"
    exit 1
}

mkdir -p "${CKPT_DIR}"


# ============================================================
# 5. Print run information
# ============================================================

echo "============================================================"
echo "Privileged RSOPD training"
echo "------------------------------------------------------------"
echo "VERL_ROOT       = ${VERL_ROOT}"
echo "STUDENT_MODEL   = ${STUDENT_MODEL}"
echo "TEACHER_MODEL   = ${TEACHER_MODEL}"
echo "TRAIN_FILE      = ${TRAIN_FILE}"
echo "VAL_FILE        = ${VAL_FILE}"
echo "PROJECT_NAME    = ${PROJECT_NAME}"
echo "EXPERIMENT_NAME = ${EXPERIMENT_NAME}"
echo "CKPT_DIR        = ${CKPT_DIR}"
echo "------------------------------------------------------------"
echo "Privileged OPD:"
echo "  enable        = true"
echo "  info_type     = full_response"
echo "============================================================"


# ============================================================
# 6. Launch
#
# Important implementation assumptions:
#
# 1. Advantage estimator:
#
#       RSOPD
#
# 2. Normal reference distribution:
#
#       ref_log_prob
#       = pi_teacher(y_t | x, y_<t)
#
# 3. Privileged teacher distribution:
#
#       teacher_log_prob
#       = pi_teacher(
#           y_t |
#           x,
#           privileged_info,
#           y_<t
#         )
#
# 4. For the current implementation:
#
#       privileged_info = full student response
#
# 5. `data.return_raw_chat=true` is REQUIRED because
#    RayPPOTrainer._build_privileged_log_prob_data()
#    reconstructs the teacher chat from `raw_prompt`.
#
# 6. The teacher model is provided through:
#
#       actor_rollout_ref.ref.model.path
#
# ============================================================

python -m verl.trainer.main_ppo \
    \
    algorithm.adv_estimator=rsopd \
    algorithm.use_kl_in_reward=False \
    algorithm.privileged_opd.teacher.enable=True \
    algorithm.privileged_opd.teacher.info_type=full_response \
    algorithm.privileged_opd.gopd_ref.enable=False \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    ++actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="${VERL_ROOT}/validation_outputs_privileged_rsopd" \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.total_epochs=1 \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations=10 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    "$@"