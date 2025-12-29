#!/bin/bash
#SBATCH --job-name=nested_cv_monitor
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# We override output/error from start_experiment.sh via sbatch options
# so no fixed #SBATCH --output/--error here.

set -euo pipefail

# ============================================================================
# Sanity checks
# ============================================================================

if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "ERROR: OUTPUT_DIR is not set. This script should be launched via start_experiment.sh."
    exit 1
fi

LOG_DIR="${LOG_DIR:-logs/default_run}"
mkdir -p "$LOG_DIR"

# Get parameters from environment (set by start_experiment.sh)
DATASET_NAME="${DATASET_NAME:-DROZY}"
SEED="${SEED:-42}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
INNER_FOLDS="${INNER_FOLDS:-3}"
N_TRIALS="${N_TRIALS:-20}"
RESUME_FLAG="${RESUME_FLAG:-}"

# Configuration
MAX_RESUBMISSIONS=50           # Maximum number of training submissions (including the first)
CHECK_INTERVAL=60              # Check every 60 seconds
TRAINING_JOB_SCRIPT="./training_job_2.sh"

# Create monitor log (I’d keep this with the experiment outputs)
MONITOR_LOG="$OUTPUT_DIR/monitor_detailed.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

# ============================================================================
# Header
# ============================================================================

echo "========================================"
echo "Monitor Job Started"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "Log dir: $LOG_DIR"
echo "========================================"

log "=========================================="
log "Monitor starting on $(hostname)"
log "Output directory: $OUTPUT_DIR"
log "Log directory: $LOG_DIR"
log "Dataset: $DATASET_NAME"
log "Configuration: $OUTER_FOLDS outer folds, $INNER_FOLDS inner folds, $N_TRIALS trials"
log "Max submissions: $MAX_RESUBMISSIONS"
log "Check interval: $CHECK_INTERVAL seconds"
log "=========================================="

# ============================================================================
# Initial training job submission (attempt #1)
# ============================================================================

log "Submitting initial training job (attempt 1/$MAX_RESUBMISSIONS)..."

if ! TRAINING_JOB_ID=$(sbatch \
    --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",LOG_DIR="$LOG_DIR",RESUME_FLAG="$RESUME_FLAG",DATASET_NAME="$DATASET_NAME",SEED="$SEED",OUTER_FOLDS="$OUTER_FOLDS",INNER_FOLDS="$INNER_FOLDS",N_TRIALS="$N_TRIALS" \
    --output="$LOG_DIR/train_%j.out" \
    --error="$LOG_DIR/train_%j.err" \
    --parsable \
    "$TRAINING_JOB_SCRIPT"); then
    log "ERROR: Failed to submit initial training job"
    touch "$OUTPUT_DIR/EXPERIMENT_FAILED"
    exit 1
fi

if [ -z "$TRAINING_JOB_ID" ]; then
    log "ERROR: sbatch returned an empty job ID for the initial training job"
    touch "$OUTPUT_DIR/EXPERIMENT_FAILED"
    exit 1
fi

log "Initial training job submitted: $TRAINING_JOB_ID"

submission_count=1
echo "Training job $submission_count: $TRAINING_JOB_ID" >> "$OUTPUT_DIR/training_job_ids.txt"

# ============================================================================
# Main monitoring loop
# ============================================================================

while true; do
    sleep "$CHECK_INTERVAL"

    # If training job is still in the queue, continue
    if squeue -j "$TRAINING_JOB_ID" 2>/dev/null | grep -q "$TRAINING_JOB_ID"; then
        continue
    fi

    log "Training job $TRAINING_JOB_ID has finished"

    RESULTS_FILE="$OUTPUT_DIR/nested_cv_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        log "=========================================="
        log "TRAINING COMPLETE!"
        log "=========================================="
        log "Results found at: $RESULTS_FILE"
        log "Total training jobs submitted: $submission_count"
        log "Monitor job completed at: $(date)"
        log "=========================================="

        touch "$OUTPUT_DIR/EXPERIMENT_COMPLETE"
        exit 0
    fi

    JOB_STATE=$(sacct -j "$TRAINING_JOB_ID" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$JOB_STATE" ]; then
        JOB_STATE="UNKNOWN"
        log "Warning: sacct did not return a state for job $TRAINING_JOB_ID. Treating as UNKNOWN."
    fi
    log "Previous job state: $JOB_STATE"

    if [ "$submission_count" -ge "$MAX_RESUBMISSIONS" ]; then
        log "=========================================="
        log "ERROR: Reached maximum submissions ($MAX_RESUBMISSIONS)."
        log "Training may have failed repeatedly. Check logs."
        log "Last training job: $TRAINING_JOB_ID"
        log "=========================================="

        touch "$OUTPUT_DIR/EXPERIMENT_FAILED"
        exit 1
    fi

    next_attempt=$((submission_count + 1))
    log "Training not complete. Resubmitting with --resume (attempt $next_attempt/$MAX_RESUBMISSIONS)..."

    if ! NEW_JOB_ID=$(sbatch \
        --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",LOG_DIR="$LOG_DIR",RESUME_FLAG="--resume",DATASET_NAME="$DATASET_NAME",SEED="$SEED",OUTER_FOLDS="$OUTER_FOLDS",INNER_FOLDS="$INNER_FOLDS",N_TRIALS="$N_TRIALS" \
        --output="$LOG_DIR/train_%j.out" \
        --error="$LOG_DIR/train_%j.err" \
        --parsable \
        "$TRAINING_JOB_SCRIPT"); then

        log "ERROR: Failed to resubmit training job on attempt $next_attempt"

        if [ "$next_attempt" -ge "$MAX_RESUBMISSIONS" ]; then
            log "Reached maximum allowed submission attempts while sbatch was failing."
            touch "$OUTPUT_DIR/EXPERIMENT_FAILED"
            exit 1
        fi

        log "Waiting 5 minutes before retry..."
        sleep 300
        continue
    fi

    if [ -z "$NEW_JOB_ID" ]; then
        log "ERROR: sbatch returned an empty job ID on attempt $next_attempt"

        if [ "$next_attempt" -ge "$MAX_RESUBMISSIONS" ]; then
            log "Reached maximum allowed submission attempts with empty job IDs."
            touch "$OUTPUT_DIR/EXPERIMENT_FAILED"
            exit 1
        fi

        log "Waiting 5 minutes before retry..."
        sleep 300
        continue
    fi

    TRAINING_JOB_ID="$NEW_JOB_ID"
    submission_count=$next_attempt

    log "Training job resubmitted: $TRAINING_JOB_ID (submission #$submission_count)"
    echo "Training job $submission_count: $TRAINING_JOB_ID" >> "$OUTPUT_DIR/training_job_ids.txt"
done