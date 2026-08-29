# Python environment used for this verl project.
VENV_ROOT="/home/wanghongye/Code/LLM/verl-cu124/.venv"

STEP=20
 
# IMPORTANT:
# Your training log showed that Python actually imported verl from:
#   /home/wanghongye/Code/verl/verl
#
# Therefore use this repo for verl.model_merger by default, so that the
# merger code is consistent with the code that produced the checkpoint.
VERL_CODE_ROOT="/home/wanghongye/Code/verl"

# Directory that contains your training checkpoints.
CKPT_ROOT="/home/wanghongye/Code/LLM/verl-cu124/checkpoints/gopd-mechanism-analysis/qwen3_4b_to_4b_rl_math_gopd_lam2_diag_all_step20_seed42"

# Source verl actor checkpoint.
LOCAL_DIR="${CKPT_ROOT}/global_step_${STEP}/actor"

# Output directory containing the merged Hugging Face model.
TARGET_DIR="/home/wanghongye/Code/LLM/verl-cu124/model/qwen3_4b_to_4b_rl_math_gopd_lam2_step${STEP}_seed42"

# Python executable.
PYTHON="${VENV_ROOT}/bin/python"


"${PYTHON}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}"