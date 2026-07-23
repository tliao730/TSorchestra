#!/bin/bash
#SBATCH --job-name=slsqp_eval
#SBATCH --array=0-96
#SBATCH --partition=gpuA100x4
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdem-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=tliao730@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p ./output/logs
source ./cli/utils.sh
activate_conda_env

# Keep HuggingFace downloads (checkpoints incl. the 2.5B Toto2) off the small
# $HOME quota — the ensemble pulls several foundation-model checkpoints.
export HF_HOME="${HF_HOME:-/work/hdd/bdem/tliao2/huggingface}"

log_info "Starting $(get_slurm_message)"

# Default to the M4 Hourly dataset (short-term) if not using SLURM
M4_HOURLY_TASK_ID=38  
DEFAULT_TASK_ID=$M4_HOURLY_TASK_ID

# Ensure SLURM_ARRAY_TASK_ID is set
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-$DEFAULT_TASK_ID}
export SLURM_ARRAY_TASK_ID

# Set run configs
metric="mae"
n_windows=1
batch_size=128
imputation="dummy_value"

if python -m pipeline.eval -cp ../conf \
    ensemble.metric="${metric}" \
    ensemble.n_windows="${n_windows}" \
    ensemble.batch_size="${batch_size}" \
    imputation="${imputation}"; then

    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exit 1
fi