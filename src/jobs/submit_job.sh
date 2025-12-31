#!/bin/bash
set -euo pipefail

# ============================================================================
# Universal Job Submission Script for HPC
# ============================================================================
# Usage: ./submit_job.sh <config_file> [--resume <run_id>]
# Example: ./submit_job.sh configs/experiment_e14.conf
# Example: ./submit_job.sh configs/experiment_e14.conf --resume 20241229_120000_E14
# ============================================================================

RESUME_MODE=false
RESUME_RUN_ID=""

# Parse arguments
if [ $# -eq 1 ]; then
    CONFIG_FILE="$1"
elif [ $# -eq 3 ] && [ "$2" = "--resume" ]; then
    CONFIG_FILE="$1"
    RESUME_MODE=true
    RESUME_RUN_ID="$3"
else
    echo "Usage: $0 <config_file> [--resume <run_id>]"
    echo ""
    echo "Examples:"
    echo "  New experiment:    $0 configs/experiment_e14.conf"
    echo "  Resume experiment: $0 configs/experiment_e14.conf --resume 20241229_120000_E14"
    exit 1
fi

# Get absolute path of config file
CONFIG_FILE=$(readlink -f "$CONFIG_FILE")

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Source the configuration
source "$CONFIG_FILE"

# Validate required variables
REQUIRED_VARS=(
    "EXPERIMENT_NAME"
    "COMMAND"
    "GPU_TIME_LIMIT"
    "MONITOR_DAYS"
    "OUTPUT_ROOT"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required variable $var is not set in $CONFIG_FILE"
        exit 1
    fi
done

# Set defaults for optional variables
CPUS_PER_TASK="${CPUS_PER_TASK:-15}"
MEM="${MEM:-40G}"
GPUS="${GPUS:-1}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
MAX_RESUBMISSIONS="${MAX_RESUBMISSIONS:-50}"
LOG_ROOT="${LOG_ROOT:-logs}"
SUCCESS_FILE="${SUCCESS_FILE:-EXPERIMENT_COMPLETE}"
ENVIRONMENT_SETUP="${ENVIRONMENT_SETUP:-}"
WORKING_DIR="${WORKING_DIR:-$(pwd)}"

# Convert paths to absolute
OUTPUT_ROOT=$(readlink -f "$OUTPUT_ROOT")
LOG_ROOT=$(readlink -f "$LOG_ROOT")
WORKING_DIR=$(readlink -f "$WORKING_DIR")

# Handle resume mode
if [ "$RESUME_MODE" = true ]; then
    # Resuming existing experiment
    RUN_ID="$RESUME_RUN_ID"
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
    LOG_DIR="${LOG_ROOT}/${RUN_ID}"
    SNAPSHOT_DIR="${OUTPUT_DIR}/code_snapshot"
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "ERROR: Cannot resume - experiment directory not found: $OUTPUT_DIR"
        exit 1
    fi
    
    if [ ! -d "$SNAPSHOT_DIR" ]; then
        echo "ERROR: Cannot resume - code snapshot not found: $SNAPSHOT_DIR"
        exit 1
    fi
    
    echo "========================================"
    echo "RESUMING Experiment: $EXPERIMENT_NAME"
    echo "========================================"
    echo "Run ID: $RUN_ID"
    echo "Output directory: $OUTPUT_DIR"
    echo "Log directory: $LOG_DIR"
    echo "Code snapshot: $SNAPSHOT_DIR"
    echo "========================================"
    echo ""
    echo "Cleaning up any existing jobs..."
    
    # Cancel any existing jobs for this experiment
    if [ -f "$OUTPUT_DIR/job_ids.txt" ]; then
        old_monitor_id=$(grep "Monitor job ID:" "$OUTPUT_DIR/job_ids.txt" | cut -d' ' -f4 || true)
        if [ -n "$old_monitor_id" ]; then
            scancel "$old_monitor_id" 2>/dev/null || echo "  Previous monitor job already finished"
        fi
    fi
    
    if [ -f "$OUTPUT_DIR/gpu_job_ids.txt" ]; then
        while read -r line; do
            job_id=$(echo "$line" | awk '{print $NF}')
            if [ -n "$job_id" ] && [[ "$job_id" =~ ^[0-9]+$ ]]; then
                scancel "$job_id" 2>/dev/null || true
            fi
        done < "$OUTPUT_DIR/gpu_job_ids.txt"
    fi
    
    # Remove failure marker if present
    rm -f "$OUTPUT_DIR/EXPERIMENT_FAILED"
    
    # Add resume entry to config
    echo "" >> "$OUTPUT_DIR/experiment_config.txt"
    echo "=== RESUMED ===" >> "$OUTPUT_DIR/experiment_config.txt"
    echo "Resume time: $(date)" >> "$OUTPUT_DIR/experiment_config.txt"
    echo "Resumed by: $USER" >> "$OUTPUT_DIR/experiment_config.txt"
    
else
    # Generate unique run ID for new experiment
    RUN_ID="$(date +%Y%m%d_%H%M%S)_${EXPERIMENT_NAME}"
    
    # Create directory structure
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
    LOG_DIR="${LOG_ROOT}/${RUN_ID}"
    SNAPSHOT_DIR="${OUTPUT_DIR}/code_snapshot"
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$SNAPSHOT_DIR"
    
    # ============================================================================
    # Create Code Snapshot (only for new experiments)
    # ============================================================================
    
    echo "Creating code snapshot..."
    
    # Copy the entire working directory (or specified paths)
    if [ -n "${SNAPSHOT_PATHS:-}" ]; then
        for path in $SNAPSHOT_PATHS; do
            if [ -e "$path" ]; then
                cp -r "$path" "$SNAPSHOT_DIR/"
            fi
        done
    else
        # Default: copy current directory, excluding common large directories
        rsync -a --exclude='.git' \
                 --exclude='__pycache__' \
                 --exclude='*.pyc' \
                 --exclude='.pytest_cache' \
                 --exclude='logs' \
                 --exclude='outputs' \
                 --exclude='.ipynb_checkpoints' \
                 --exclude='*.egg-info' \
                 "$WORKING_DIR/" "$SNAPSHOT_DIR/"
    fi
    
    # Save git information if in a git repo
    if git rev-parse --git-dir > /dev/null 2>&1; then
        git rev-parse HEAD > "$SNAPSHOT_DIR/git_commit.txt" 2>/dev/null || true
        git diff > "$SNAPSHOT_DIR/git_diff.txt" 2>/dev/null || true
        git status > "$SNAPSHOT_DIR/git_status.txt" 2>/dev/null || true
    fi
    
    # ============================================================================
    # Save Configuration
    # ============================================================================
    
    cat > "$OUTPUT_DIR/experiment_config.txt" << EOF
Experiment started: $(date)
Run ID: $RUN_ID
Experiment name: $EXPERIMENT_NAME
Output directory: $OUTPUT_DIR
Log directory: $LOG_DIR
Code snapshot: $SNAPSHOT_DIR
Working directory: $WORKING_DIR

=== Job Configuration ===
GPU time limit: $GPU_TIME_LIMIT
Monitor duration: $MONITOR_DAYS days
CPUs per task: $CPUS_PER_TASK
Memory: $MEM
GPUs: $GPUS
Max resubmissions: $MAX_RESUBMISSIONS
Check interval: $CHECK_INTERVAL seconds

=== Command ===
$COMMAND

=== Environment Setup ===
${ENVIRONMENT_SETUP:-None}

=== Config File ===
EOF
    
    cat "$CONFIG_FILE" >> "$OUTPUT_DIR/experiment_config.txt"
    
    # Copy the config file for reference
    cp "$CONFIG_FILE" "$OUTPUT_DIR/original_config.conf"
    
    echo "========================================"
    echo "Starting Experiment: $EXPERIMENT_NAME"
    echo "========================================"
    echo "Run ID: $RUN_ID"
    echo "Output directory: $OUTPUT_DIR"
    echo "Log directory: $LOG_DIR"
    echo "Code snapshot: $SNAPSHOT_DIR"
    echo "Monitor duration: $MONITOR_DAYS days"
    echo "========================================"
fi

# ============================================================================
# Submit Monitor Job
# ============================================================================

echo ""
echo "Submitting monitor job..."

MONITOR_TIME="${MONITOR_DAYS}-00:00:00"

# Create temporary monitor script
MONITOR_SCRIPT=$(mktemp)
cat > "$MONITOR_SCRIPT" << 'MONITOR_EOF'
#!/bin/bash
#SBATCH --job-name=monitor
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

MONITOR_LOG="$OUTPUT_DIR/monitor_detailed.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

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
log "Experiment: $EXPERIMENT_NAME"
log "Max submissions: $MAX_RESUBMISSIONS"
log "Check interval: $CHECK_INTERVAL seconds"
log "=========================================="

# Create GPU job script
GPU_JOB_SCRIPT="$OUTPUT_DIR/gpu_job.sh"
cat > "$GPU_JOB_SCRIPT" << 'GPU_EOF'
#!/bin/bash
#SBATCH --job-name=gpu_task
#SBATCH --gres=gpu:GPU_PLACEHOLDER
#SBATCH --cpus-per-task=CPU_PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER

set -euo pipefail

echo "========================================"
echo "GPU Job Started"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

# Change to snapshot directory
cd "$SNAPSHOT_DIR"

# Setup environment
ENV_SETUP

# Run the command
COMMAND_PLACEHOLDER

exit_code=$?

echo "========================================"
echo "GPU Job Finished"
echo "========================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "========================================"

exit $exit_code
GPU_EOF

# Replace placeholders
sed -i "s|GPU_PLACEHOLDER|$GPUS|g" "$GPU_JOB_SCRIPT"
sed -i "s|CPU_PLACEHOLDER|$CPUS_PER_TASK|g" "$GPU_JOB_SCRIPT"
sed -i "s|MEM_PLACEHOLDER|$MEM|g" "$GPU_JOB_SCRIPT"
sed -i "s|TIME_PLACEHOLDER|$GPU_TIME_LIMIT|g" "$GPU_JOB_SCRIPT"

# Replace ENV_SETUP (handle special characters and newlines)
if [ -n "$ENVIRONMENT_SETUP" ]; then
    # Escape special characters for sed
    ENV_SETUP_ESCAPED=$(printf '%s\n' "$ENVIRONMENT_SETUP" | sed -e 's/[\/&]/\\&/g')
    sed -i "s|ENV_SETUP|$ENV_SETUP_ESCAPED|g" "$GPU_JOB_SCRIPT"
else
    sed -i "s|ENV_SETUP|echo 'No environment setup'|g" "$GPU_JOB_SCRIPT"
fi

# Replace COMMAND_PLACEHOLDER (handle special characters)
COMMAND_ESCAPED=$(printf '%s\n' "$COMMAND" | sed -e 's/[\/&]/\\&/g')
sed -i "s|COMMAND_PLACEHOLDER|$COMMAND_ESCAPED|g" "$GPU_JOB_SCRIPT"

chmod +x "$GPU_JOB_SCRIPT"

# Submit initial GPU job
log "Submitting initial GPU job (attempt 1/$MAX_RESUBMISSIONS)..."

if ! GPU_JOB_ID=$(sbatch \
    --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SNAPSHOT_DIR="$SNAPSHOT_DIR" \
    --output="$LOG_DIR/gpu_%j.out" \
    --error="$LOG_DIR/gpu_%j.err" \
    --parsable \
    "$GPU_JOB_SCRIPT"); then
    log "ERROR: Failed to submit initial GPU job"
    touch "$OUTPUT_DIR/EXPERIMENT_FAILED"
    exit 1
fi

log "Initial GPU job submitted: $GPU_JOB_ID"

submission_count=1
echo "GPU job $submission_count: $GPU_JOB_ID" >> "$OUTPUT_DIR/gpu_job_ids.txt"

# Main monitoring loop
while true; do
    sleep "$CHECK_INTERVAL"

    # Check if job is still running
    if squeue -j "$GPU_JOB_ID" 2>/dev/null | grep -q "$GPU_JOB_ID"; then
        continue
    fi

    log "GPU job $GPU_JOB_ID has finished"

    # Check for success
    if [ -f "$OUTPUT_DIR/$SUCCESS_FILE" ]; then
        log "=========================================="
        log "EXPERIMENT COMPLETE!"
        log "=========================================="
        log "Success file found: $SUCCESS_FILE"
        log "Total GPU jobs submitted: $submission_count"
        log "Monitor job completed at: $(date)"
        log "=========================================="
        exit 0
    fi

    # Get job state
    JOB_STATE=$(sacct -j "$GPU_JOB_ID" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$JOB_STATE" ]; then
        JOB_STATE="UNKNOWN"
    fi
    log "Previous job state: $JOB_STATE"

    # Check if we've hit the max
    if [ "$submission_count" -ge "$MAX_RESUBMISSIONS" ]; then
        log "=========================================="
        log "ERROR: Reached maximum submissions ($MAX_RESUBMISSIONS)."
        log "Last GPU job: $GPU_JOB_ID"
        log "=========================================="
        touch "$OUTPUT_DIR/EXPERIMENT_FAILED"
        exit 1
    fi

    # Resubmit
    next_attempt=$((submission_count + 1))
    log "Resubmitting GPU job (attempt $next_attempt/$MAX_RESUBMISSIONS)..."

    if ! NEW_JOB_ID=$(sbatch \
        --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SNAPSHOT_DIR="$SNAPSHOT_DIR" \
        --output="$LOG_DIR/gpu_%j.out" \
        --error="$LOG_DIR/gpu_%j.err" \
        --parsable \
        "$GPU_JOB_SCRIPT"); then
        log "ERROR: Failed to resubmit GPU job on attempt $next_attempt"
        
        if [ "$next_attempt" -ge "$MAX_RESUBMISSIONS" ]; then
            touch "$OUTPUT_DIR/EXPERIMENT_FAILED"
            exit 1
        fi
        
        log "Waiting 5 minutes before retry..."
        sleep 300
        continue
    fi

    GPU_JOB_ID="$NEW_JOB_ID"
    submission_count=$next_attempt

    log "GPU job resubmitted: $GPU_JOB_ID (submission #$submission_count)"
    echo "GPU job $submission_count: $GPU_JOB_ID" >> "$OUTPUT_DIR/gpu_job_ids.txt"
done
MONITOR_EOF

chmod +x "$MONITOR_SCRIPT"

# Submit monitor job
MONITOR_JOB_ID=$(sbatch \
    --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",LOG_DIR="$LOG_DIR",SNAPSHOT_DIR="$SNAPSHOT_DIR",EXPERIMENT_NAME="$EXPERIMENT_NAME",MAX_RESUBMISSIONS="$MAX_RESUBMISSIONS",CHECK_INTERVAL="$CHECK_INTERVAL",GPU_TIME_LIMIT="$GPU_TIME_LIMIT",CPUS_PER_TASK="$CPUS_PER_TASK",MEM="$MEM",GPUS="$GPUS",SUCCESS_FILE="$SUCCESS_FILE",ENVIRONMENT_SETUP="$ENVIRONMENT_SETUP",COMMAND="$COMMAND" \
    --time="$MONITOR_TIME" \
    --output="$LOG_DIR/monitor_%j.out" \
    --error="$LOG_DIR/monitor_%j.err" \
    --parsable \
    "$MONITOR_SCRIPT")

# Cleanup temp script
rm "$MONITOR_SCRIPT"

if [ -z "$MONITOR_JOB_ID" ]; then
    echo "ERROR: Failed to submit monitor job"
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
echo "Log directory: $LOG_DIR"
echo "Code snapshot: $SNAPSHOT_DIR"
echo ""
echo "To check status:"
echo "  squeue -j $MONITOR_JOB_ID"
echo ""
echo "To view monitor log:"
echo "  tail -f $LOG_DIR/monitor_${MONITOR_JOB_ID}.out"
echo "  tail -f $OUTPUT_DIR/monitor_detailed.log"
echo ""
echo "To cancel:"
echo "  scancel $MONITOR_JOB_ID"
echo ""
echo "Results will be in:"
echo "  $OUTPUT_DIR/"
echo "========================================"

echo "Monitor job ID: $MONITOR_JOB_ID" > "$OUTPUT_DIR/job_ids.txt"
echo "Started: $(date)" >> "$OUTPUT_DIR/job_ids.txt"