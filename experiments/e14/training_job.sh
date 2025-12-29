#!/bin/bash
#SBATCH --job-name=nested_cv_train
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================================
# Training Job - Runs on GPU node for 24 hours max
# ============================================================================

set -euo pipefail

if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "ERROR: OUTPUT_DIR is not set. This script is meant to be launched via start_experiment.sh."
    exit 1
fi

echo "========================================"
echo "Training Job Started"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "Resume flag: $RESUME_FLAG"
echo "========================================"

# Activate environment
cd /home/mojtaba.kananisarche/
module load conda
conda activate .cvenv

# Get parameters from environment
DATASET_NAME="${DATASET_NAME:-DROZY}"
SEED="${SEED:-42}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
INNER_FOLDS="${INNER_FOLDS:-3}"
N_TRIALS="${N_TRIALS:-20}"

# Run training
python -m Zh.experiments.e14.main \
    --dataset_name "$DATASET_NAME" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    $RESUME_FLAG \
    --parallel_inner_cv \
    --max_workers 6 \
    --cv.outer_folds "$OUTER_FOLDS" \
    --cv.inner_folds "$INNER_FOLDS" \
    --cv.search_strategy random \
    --cv.n_trials "$N_TRIALS" \
    --train.num_epochs 200 \
    --train.batch_size 32 \
    --train.device cuda

exit_code=$?

echo "========================================"
echo "Training Job Finished"
echo "========================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "========================================"

exit $exit_code