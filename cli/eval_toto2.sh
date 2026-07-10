#!/bin/bash
#SBATCH --job-name=toto2_eval
#SBATCH --array=0-96
#SBATCH --partition=gpuA40x4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdem-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=tliao730@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Standalone GIFT-Eval scoring sweep for Toto 2.0 (Datadog/Toto-2.0-2.5B-FT).
# One array task per dataset/term config (97 total, indices 0-96 into
# conf/data/dataset.yaml). Results land in results/Toto2/<dataset_config>/.
#
# Single-config test run (M4 Hourly, short-term) without SLURM:
#   bash cli/eval_toto2.sh
# Full sweep:
#   sbatch cli/eval_toto2.sh

mkdir -p ./output/logs
source ./cli/utils.sh

# Activate the tso env via the miniforge3 module's conda (envs live in
# /work/nvme/bcqc/tliao2/conda/envs per ~/.condarc — /projects/bdem is over
# its inode quota and cannot hold a conda env).
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate tso

# Keep HuggingFace downloads (checkpoints, datasets) off the small $HOME quota
export HF_HOME="${HF_HOME:-/work/hdd/bdem/tliao2/huggingface}"

log_info "Starting $(get_slurm_message)"

# Default to the M4 Hourly dataset (short-term) if not using SLURM
M4_HOURLY_TASK_ID=38
DEFAULT_TASK_ID=$M4_HOURLY_TASK_ID

# Ensure SLURM_ARRAY_TASK_ID is set
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-$DEFAULT_TASK_ID}
export SLURM_ARRAY_TASK_ID

# Set run configs. batch_size is passed to both the GluonTS predictor and the
# Toto2 model itself — 32 is conservative for the 2.5B checkpoint on an A40
# (48 GB); raise it if GPU memory allows, lower it if long-horizon configs OOM.
batch_size=32
imputation="dummy_value"

if python -m pipeline.eval_toto2 -cp ../conf \
    batch_size="${batch_size}" \
    imputation="${imputation}"; then

    log_info "Successfully finished $(get_slurm_message)!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"

    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exit 1
fi
