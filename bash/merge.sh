# Python environment used for this verl project.
VENV_ROOT="/home/wanghongye/Code/LLM/verl-cu124/.venv"

# IMPORTANT:
# Your training log showed that Python actually imported verl from:
#   /home/wanghongye/Code/verl/verl
#
# Therefore use this repo for verl.model_merger by default, so that the
# merger code is consistent with the code that produced the checkpoint.
VERL_CODE_ROOT="/home/wanghongye/Code/verl"

# Directory that contains your training checkpoints.
CKPT_ROOT="/home/wanghongye/Code/LLM/verl-cu124/checkpoints/rsopd_math/qwen3_1.7b_to_4b_math1500_seed42"

# Source verl actor checkpoint.
LOCAL_DIR="${CKPT_ROOT}/global_step_${STEP}/actor"

# Output directory containing the merged Hugging Face model.
TARGET_DIR="${LOCAL_DIR}/merged_hf"

# Python executable.
PYTHON="${VENV_ROOT}/bin/python"


"${PYTHON}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}"