#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# G-OPD pilot / correctness training
#
#   Student      : Qwen3-1.7B
#   Teacher      : Qwen3-4B
#   G-OPD ref    : student initial checkpoint by default
#   Lambda       : 1.0 by default (should reduce to standard OPD)
#   Train        : 1500 stratified MATH examples
#   Eval         : full MATH-500
#   GPUs         : 4 x H100
#
# This script also supports independently enabling privileged
# information for the teacher and the G-OPD reference:
#
#   PRIVILEGED_TEACHER=true
#   TEACHER_INFO_TYPE=full_response
#
#   PRIVILEGED_GOPD_REF=true
#   GOPD_REF_INFO_TYPE=full_response
#
# Examples:
#
#   # 1) Correctness test: lambda = 1.0, no privileged context
#   bash run_gopd_test.sh
#
#   # 2) ExOPD-style test
#   GOPD_LAM=1.25 bash run_gopd_test.sh
#
#   # 3) Privileged teacher only
#   GOPD_LAM=1.25 \
#   PRIVILEGED_TEACHER=true \
#   bash run_gopd_test.sh
#
#   # 4) Privileged G-OPD reference only
#   GOPD_LAM=1.25 \
#   PRIVILEGED_GOPD_REF=true \
#   bash run_gopd_test.sh
#
#   # 5) Both independently privileged
#   GOPD_LAM=1.25 \
#   PRIVILEGED_TEACHER=true \
#   PRIVILEGED_GOPD_REF=true \
#   bash run_gopd_test.sh
# ============================================================


# ============================================================
# 0. Environment
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"


# ============================================================
# 1. Project paths and models
# ============================================================

VERL_ROOT="${VERL_ROOT:-$HOME/Code/LLM/verl-cu124}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${VERL_ROOT}/swanlog}"

STUDENT_MODEL="${STUDENT_MODEL:-${VERL_ROOT}/model/Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-${VERL_ROOT}/model/Qwen3-4B}"

# Vanilla G-OPD uses the student's initial checkpoint as pi_R.
# Override GOPD_REF_MODEL if you deliberately want another
# reference model.
GOPD_REF_MODEL="${GOPD_REF_MODEL:-${STUDENT_MODEL}}"

TRAIN_FILE="${TRAIN_FILE:-${VERL_ROOT}/data/MATH/math_train_1500_verl.parquet}"
VAL_FILE="${VAL_FILE:-${VERL_ROOT}/data/MATH-500/math500_verl.parquet}"


# ============================================================
# 2. G-OPD / privileged configuration
# ============================================================

# Start with lambda=1.0 as a correctness test:
# G-OPD should reduce to standard OPD.
GOPD_LAM="${GOPD_LAM:-1.25}"

PRIVILEGED_TEACHER="${PRIVILEGED_TEACHER:-false}"
TEACHER_INFO_TYPE="${TEACHER_INFO_TYPE:-full_response}"

PRIVILEGED_GOPD_REF="${PRIVILEGED_GOPD_REF:-false}"
GOPD_REF_INFO_TYPE="${GOPD_REF_INFO_TYPE:-full_response}"

case "${PRIVILEGED_TEACHER}" in
    true|false) ;;
    *)
        echo "PRIVILEGED_TEACHER must be 'true' or 'false'."
        exit 1
        ;;
esac

case "${PRIVILEGED_GOPD_REF}" in
    true|false) ;;
    *)
        echo "PRIVILEGED_GOPD_REF must be 'true' or 'false'."
        exit 1
        ;;
esac

if [[ "${PRIVILEGED_TEACHER}" == "true" || \
      "${PRIVILEGED_GOPD_REF}" == "true" ]]; then
    RETURN_RAW_CHAT=true
else
    RETURN_RAW_CHAT=false
fi

MODE_TAG="standard"
if [[ "${PRIVILEGED_TEACHER}" == "true" && \
      "${PRIVILEGED_GOPD_REF}" == "true" ]]; then
    MODE_TAG="priv_teacher_priv_ref"
elif [[ "${PRIVILEGED_TEACHER}" == "true" ]]; then
    MODE_TAG="priv_teacher"
elif [[ "${PRIVILEGED_GOPD_REF}" == "true" ]]; then
    MODE_TAG="priv_ref"
fi


# ============================================================
# 3. Experiment naming
# ============================================================

PROJECT_NAME="${PROJECT_NAME:-gopd_math}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1.7b_to_4b_math1500_lam${GOPD_LAM}_${MODE_TAG}_seed42}"
CKPT_DIR="${CKPT_DIR:-${VERL_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
VAL_OUTPUT_DIR="${VAL_OUTPUT_DIR:-${VERL_ROOT}/validation_outputs_gopd/${EXPERIMENT_NAME}}"

cd "${VERL_ROOT}"


# ============================================================
# 4. Dependency and file checks
# ============================================================

python -c "import swanlab" >/dev/null 2>&1 || {
    echo "SwanLab is not installed in the current Python environment."
    echo 'Install it with: uv pip install --python "$VIRTUAL_ENV/bin/python" swanlab'
    exit 1
}

test -d "${STUDENT_MODEL}" || {
    echo "Student model not found: ${STUDENT_MODEL}"
    exit 1
}

test -d "${TEACHER_MODEL}" || {
    echo "Teacher model not found: ${TEACHER_MODEL}"
    exit 1
}

test -d "${GOPD_REF_MODEL}" || {
    echo "G-OPD reference model not found: ${GOPD_REF_MODEL}"
    exit 1
}

test -f "${TRAIN_FILE}" || {
    echo "Train parquet not found: ${TRAIN_FILE}"
    exit 1
}

test -f "${VAL_FILE}" || {
    echo "Validation parquet not found: ${VAL_FILE}"
    exit 1
}

mkdir -p "${CKPT_DIR}"
mkdir -p "${VAL_OUTPUT_DIR}"


# ============================================================
# 5. Print run information
# ============================================================

echo "============================================================"
echo "G-OPD training"
echo "------------------------------------------------------------"
echo "VERL_ROOT             = ${VERL_ROOT}"
echo "STUDENT_MODEL         = ${STUDENT_MODEL}"
echo "TEACHER_MODEL         = ${TEACHER_MODEL}"
echo "GOPD_REF_MODEL        = ${GOPD_REF_MODEL}"
echo "GOPD_LAM              = ${GOPD_LAM}"
echo "TRAIN_FILE            = ${TRAIN_FILE}"
echo "VAL_FILE              = ${VAL_FILE}"
echo "PROJECT_NAME          = ${PROJECT_NAME}"
echo "EXPERIMENT_NAME       = ${EXPERIMENT_NAME}"
echo "CKPT_DIR              = ${CKPT_DIR}"
echo "------------------------------------------------------------"
echo "Privileged teacher:"
echo "  enable              = ${PRIVILEGED_TEACHER}"
echo "  info_type           = ${TEACHER_INFO_TYPE}"
echo "Privileged G-OPD ref:"
echo "  enable              = ${PRIVILEGED_GOPD_REF}"
echo "  info_type           = ${GOPD_REF_INFO_TYPE}"
echo "data.return_raw_chat  = ${RETURN_RAW_CHAT}"
echo "============================================================"


# ============================================================
# 6. Launch
#
# Expected G-OPD tensors:
#
#   old_log_probs
#       = log pi_S(y_t | x, y_<t)
#
#   ref_log_prob / teacher_log_prob
#       = log pi_T(y_t | context)
#
#   gopd_ref_log_prob
#       = log pi_R(y_t | context)
#
# Advantage:
#
#   A_t =
#       lambda * (log pi_T - log pi_R)
#       - (log pi_S - log pi_R)
#
# With lambda=1:
#
#   A_t = log pi_T - log pi_S
#
# so it should reduce to standard RSOPD/OPD.
# ============================================================

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gopd \
    algorithm.use_kl_in_reward=False \
    algorithm.gopd.lam="${GOPD_LAM}" \
    algorithm.privileged_opd.teacher.enable="${PRIVILEGED_TEACHER}" \
    algorithm.privileged_opd.teacher.info_type="${TEACHER_INFO_TYPE}" \
    algorithm.privileged_opd.gopd_ref.enable="${PRIVILEGED_GOPD_REF}" \
    algorithm.privileged_opd.gopd_ref.info_type="${GOPD_REF_INFO_TYPE}" \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.return_raw_chat="${RETURN_RAW_CHAT}" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
    actor_rollout_ref.gopd_ref.model.path="${GOPD_REF_MODEL}" \
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
    trainer.validation_data_dir="${VAL_OUTPUT_DIR}" \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.total_epochs=1 \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations=10 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    "$@"
