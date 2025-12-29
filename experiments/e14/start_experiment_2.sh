#!/bin/bash
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Unique run ID for this experiment
# RUN_ID="$(date +%Y%m%d_%H%M%S)_E14"
RUN_ID="20251228_202300_E14"

# Output and log directories for this run
OUTPUT_ROOT="/home/mojtaba.kananisarche/Zh/outputs"
LOG_ROOT="logs"

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"

# Dataset to use
DATASET_NAME="DROZY"

# Random seed
SEED=24

# CV configuration
OUTER_FOLDS=5
INNER_FOLDS=4
N_TRIALS=12

# How long should the monitor run? (days)
MONITOR_DAYS=7
RESUME_FLAG="--resume"

# ============================================================================
# Setup
# ============================================================================

echo "========================================"
echo "Starting Nested CV Experiment"
echo "========================================"
echo "Run ID:          $RUN_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory:    $LOG_DIR"
echo "Dataset:          $DATASET_NAME"
echo "Monitor duration: $MONITOR_DAYS days"
echo "========================================"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

cat > "$OUTPUT_DIR/experiment_config.txt" << EOF
Experiment started: $(date)
Run ID: $RUN_ID
Output directory: $OUTPUT_DIR
Log directory: $LOG_DIR
Dataset: $DATASET_NAME
Seed: $SEED
Outer folds: $OUTER_FOLDS
Inner folds: $INNER_FOLDS
N trials: $N_TRIALS
Monitor duration: $MONITOR_DAYS days
EOF

# ============================================================================
# Submit Monitor Job
# ============================================================================

echo "Submitting monitor job..."

MONITOR_TIME="${MONITOR_DAYS}-00:00:00"
echo "Time limit: $MONITOR_TIME"

MONITOR_JOB_ID=$(sbatch \
    --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",LOG_DIR="$LOG_DIR",DATASET_NAME="$DATASET_NAME",SEED="$SEED",OUTER_FOLDS="$OUTER_FOLDS",INNER_FOLDS="$INNER_FOLDS",N_TRIALS="$N_TRIALS",RESUME_FLAG="$RESUME_FLAG" \
    --time="$MONITOR_TIME" \
    --output="$LOG_DIR/monitor_%j.out" \
    --error="$LOG_DIR/monitor_%j.err" \
    --parsable \
    monitor_job_2.sh)

if [ -z "$MONITOR_JOB_ID" ]; then
    echo "ERROR: Failed to submit monitor job"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check available partitions: sinfo"
    echo "2. Check if time limit is too long: scontrol show partition"
    echo "3. Try reducing MONITOR_DAYS (currently: $MONITOR_DAYS)"
    echo "4. Check monitor_job.sh has no conflicting #SBATCH directives"
    exit 1
fi

echo "✓ Monitor job submitted: $MONITOR_JOB_ID"
echo ""
echo "========================================"
echo "Experiment Started Successfully!"
echo "========================================"
echo ""
echo "Monitor job ID: $MONITOR_JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory:    $LOG_DIR"
echo ""
echo "To check status:"
echo "  squeue -j $MONITOR_JOB_ID"
echo ""
echo "To view monitor log:"
echo "  tail -f $LOG_DIR/monitor_${MONITOR_JOB_ID}.out"
echo ""
echo "To cancel:"
echo "  scancel $MONITOR_JOB_ID"
echo ""
echo "Results will be in:"
echo "  $OUTPUT_DIR/nested_cv_results.json"
echo "========================================"

echo "Monitor job ID: $MONITOR_JOB_ID" > "$OUTPUT_DIR/job_ids.txt"
echo "Started: $(date)" >> "$OUTPUT_DIR/job_ids.txt"