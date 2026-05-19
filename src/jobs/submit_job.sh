#!/bin/bash
set -euo pipefail

# ============================================================================
# Universal Job Submission Script for HPC
# Supports CPU or GPU jobs via config, plus include/exclude partitions and nodes
# Usage: ./submit_job.sh <config_file> [--resume <run_id>]
# ============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    echo "ERROR: This script must be run with bash."
    exit 1
fi

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
    exit 1
fi

CONFIG_FILE=$(readlink -f "$CONFIG_FILE")

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

REQUIRED_VARS=(
    "EXPERIMENT_NAME"
    "COMMAND"
    "OUTPUT_ROOT"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required variable $var is not set in $CONFIG_FILE"
        exit 1
    fi
done

# Defaults
CPUS_PER_TASK="${CPUS_PER_TASK:-15}"
MEM="${MEM:-40G}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
MAX_RESUBMISSIONS="${MAX_RESUBMISSIONS:-50}"
SUCCESS_FILE="${SUCCESS_FILE:-EXPERIMENT_COMPLETE}"
ENVIRONMENT_SETUP="${ENVIRONMENT_SETUP:-}"
WORKING_DIR="${WORKING_DIR:-$(pwd)}"

# CPU/GPU mode
RESOURCE_TYPE="${RESOURCE_TYPE:-gpu}"      # gpu or cpu
GPUS="${GPUS:-1}"                         # used only if RESOURCE_TYPE=gpu
GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-24:00:00}"
CPU_TIME_LIMIT="${CPU_TIME_LIMIT:-24:00:00}"
JOB_TIME_LIMIT="${JOB_TIME_LIMIT:-}"      # optional override

if [ -n "$JOB_TIME_LIMIT" ]; then
    TIME_LIMIT="$JOB_TIME_LIMIT"
else
    if [ "$RESOURCE_TYPE" = "cpu" ]; then
        TIME_LIMIT="$CPU_TIME_LIMIT"
    else
        TIME_LIMIT="$GPU_TIME_LIMIT"
    fi
fi

# Partition and node controls (legacy/global defaults)
INCLUDE_PARTITIONS="${INCLUDE_PARTITIONS:-}"   # comma or space separated
EXCLUDE_PARTITIONS="${EXCLUDE_PARTITIONS:-}"   # comma or space separated
EXCLUDE_NODES="${EXCLUDE_NODES:-}"             # comma separated node list

# New: Partition scoping controls
# Modes: inherit | custom | unrestricted
# - inherit: uses INCLUDE_PARTITIONS / EXCLUDE_PARTITIONS
# - custom: uses job-specific INCLUDE/EXCLUDE variables below
# - unrestricted: does not pass --partition for that job
COMPUTE_PARTITION_MODE="${COMPUTE_PARTITION_MODE:-inherit}"
MONITOR_PARTITION_MODE="${MONITOR_PARTITION_MODE:-inherit}"

COMPUTE_INCLUDE_PARTITIONS="${COMPUTE_INCLUDE_PARTITIONS:-}"
COMPUTE_EXCLUDE_PARTITIONS="${COMPUTE_EXCLUDE_PARTITIONS:-}"
MONITOR_INCLUDE_PARTITIONS="${MONITOR_INCLUDE_PARTITIONS:-}"
MONITOR_EXCLUDE_PARTITIONS="${MONITOR_EXCLUDE_PARTITIONS:-}"

# Optional scheduling controls
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"

OUTPUT_ROOT=$(readlink -f "$OUTPUT_ROOT")
WORKING_DIR=$(readlink -f "$WORKING_DIR")

_normalize_list_to_lines() {
    local s="${1:-}"
    if [ -z "$s" ]; then
        return 0
    fi
    echo "$s" | tr ', ' '\n' | sed '/^$/d'
}

_get_all_partitions() {
    sinfo -h -o "%P" 2>/dev/null | sed 's/\*$//' | sed '/^$/d' | sort -u
}

_build_partition_csv() {
    local include_raw="${1:-}"
    local exclude_raw="${2:-}"

    local include_list=""
    local exclude_list=""
    include_list="$(_normalize_list_to_lines "$include_raw" | sort -u || true)"
    exclude_list="$(_normalize_list_to_lines "$exclude_raw" | sort -u || true)"

    local base_list=""
    if [ -n "$include_list" ]; then
        base_list="$include_list"
    else
        base_list="$(_get_all_partitions)"
    fi

    if [ -z "$base_list" ]; then
        echo ""
        return 0
    fi

    local final_list=""
    if [ -n "$exclude_list" ]; then
        while IFS= read -r p; do
            [ -z "$p" ] && continue
            if echo "$exclude_list" | grep -qxF "$p"; then
                continue
            fi
            final_list+="${p}"$'\n'
        done <<< "$base_list"
        final_list="$(echo "$final_list" | sed '/^$/d' | sort -u)"
    else
        final_list="$(echo "$base_list" | sed '/^$/d' | sort -u)"
    fi

    if [ -z "$final_list" ]; then
        echo ""
        return 0
    fi

    echo "$final_list" | paste -sd, -
}

_resolve_partition_csv_for_mode() {
    local mode="${1:-inherit}"
    local include_raw="${2:-}"
    local exclude_raw="${3:-}"

    case "$mode" in
        unrestricted)
            echo ""
            return 0
            ;;
        inherit)
            _build_partition_csv "$INCLUDE_PARTITIONS" "$EXCLUDE_PARTITIONS"
            return 0
            ;;
        custom)
            _build_partition_csv "$include_raw" "$exclude_raw"
            return 0
            ;;
        *)
            echo "ERROR: Invalid partition mode: '$mode' (expected inherit|custom|unrestricted)"
            return 1
            ;;
    esac
}

COMPUTE_PARTITION_CSV="$(_resolve_partition_csv_for_mode "$COMPUTE_PARTITION_MODE" "$COMPUTE_INCLUDE_PARTITIONS" "$COMPUTE_EXCLUDE_PARTITIONS")"
MONITOR_PARTITION_CSV="$(_resolve_partition_csv_for_mode "$MONITOR_PARTITION_MODE" "$MONITOR_INCLUDE_PARTITIONS" "$MONITOR_EXCLUDE_PARTITIONS")"

# If compute is not unrestricted, ensure it did not become an empty set
if [ "$COMPUTE_PARTITION_MODE" != "unrestricted" ] && [ -z "$COMPUTE_PARTITION_CSV" ]; then
    echo "ERROR: Compute partition selection resulted in an empty set."
    echo "  COMPUTE_PARTITION_MODE='$COMPUTE_PARTITION_MODE'"
    echo "  INCLUDE_PARTITIONS='$INCLUDE_PARTITIONS'"
    echo "  EXCLUDE_PARTITIONS='$EXCLUDE_PARTITIONS'"
    echo "  COMPUTE_INCLUDE_PARTITIONS='$COMPUTE_INCLUDE_PARTITIONS'"
    echo "  COMPUTE_EXCLUDE_PARTITIONS='$COMPUTE_EXCLUDE_PARTITIONS'"
    exit 1
fi

# If monitor is not unrestricted, ensure it did not become an empty set
if [ "$MONITOR_PARTITION_MODE" != "unrestricted" ] && [ -z "$MONITOR_PARTITION_CSV" ]; then
    echo "ERROR: Monitor partition selection resulted in an empty set."
    echo "  MONITOR_PARTITION_MODE='$MONITOR_PARTITION_MODE'"
    echo "  INCLUDE_PARTITIONS='$INCLUDE_PARTITIONS'"
    echo "  EXCLUDE_PARTITIONS='$EXCLUDE_PARTITIONS'"
    echo "  MONITOR_INCLUDE_PARTITIONS='$MONITOR_INCLUDE_PARTITIONS'"
    echo "  MONITOR_EXCLUDE_PARTITIONS='$MONITOR_EXCLUDE_PARTITIONS'"
    exit 1
fi

SBATCH_PARTITION_OPT_COMPUTE=""
if [ -n "$COMPUTE_PARTITION_CSV" ]; then
    SBATCH_PARTITION_OPT_COMPUTE="--partition=${COMPUTE_PARTITION_CSV}"
fi

SBATCH_PARTITION_OPT_MONITOR=""
if [ -n "$MONITOR_PARTITION_CSV" ]; then
    SBATCH_PARTITION_OPT_MONITOR="--partition=${MONITOR_PARTITION_CSV}"
fi

SBATCH_EXCLUDE_OPT=""
if [ -n "$EXCLUDE_NODES" ]; then
    SBATCH_EXCLUDE_OPT="--exclude=${EXCLUDE_NODES}"
fi

SBATCH_ACCOUNT_OPT=""
if [ -n "$SBATCH_ACCOUNT" ]; then
    SBATCH_ACCOUNT_OPT="--account=${SBATCH_ACCOUNT}"
fi

SBATCH_QOS_OPT=""
if [ -n "$SBATCH_QOS" ]; then
    SBATCH_QOS_OPT="--qos=${SBATCH_QOS}"
fi

SBATCH_CONSTRAINT_OPT=""
if [ -n "$SBATCH_CONSTRAINT" ]; then
    SBATCH_CONSTRAINT_OPT="--constraint=${SBATCH_CONSTRAINT}"
fi

# ----------------------------------------------------------------------------
# Resume vs new experiment
# ----------------------------------------------------------------------------
if [ "$RESUME_MODE" = true ]; then
    RUN_ID="$RESUME_RUN_ID"
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
    MONITOR_DIR="${OUTPUT_DIR}/monitor"
    LOG_DIR="${OUTPUT_DIR}/logs"
    SNAPSHOT_DIR="${MONITOR_DIR}/code_snapshot"

    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "ERROR: Cannot resume, experiment directory not found: $OUTPUT_DIR"
        exit 1
    fi
    if [ ! -d "$SNAPSHOT_DIR" ]; then
        echo "ERROR: Cannot resume, code snapshot not found: $SNAPSHOT_DIR"
        exit 1
    fi

    echo "Resuming: $RUN_ID"
    echo "Resource: $RESOURCE_TYPE"
    echo "Compute partitions: ${COMPUTE_PARTITION_CSV:-None}"
    echo "Monitor partitions: ${MONITOR_PARTITION_CSV:-None}"
    echo "Exclude nodes: ${EXCLUDE_NODES:-None}"

    if [ -f "$MONITOR_DIR/job_ids.txt" ]; then
        old_monitor_id=$(grep "Monitor job ID:" "$MONITOR_DIR/job_ids.txt" | cut -d' ' -f4 || true)
        if [ -n "$old_monitor_id" ]; then
            scancel "$old_monitor_id" 2>/dev/null || true
        fi
    fi

    if [ -f "$MONITOR_DIR/gpu_job_ids.txt" ]; then
        while read -r line; do
            job_id=$(echo "$line" | awk '{print $NF}')
            if [ -n "$job_id" ] && [[ "$job_id" =~ ^[0-9]+$ ]]; then
                scancel "$job_id" 2>/dev/null || true
            fi
        done < "$MONITOR_DIR/gpu_job_ids.txt"
    fi

    rm -f "$MONITOR_DIR/EXPERIMENT_FAILED"

else
    RUN_ID="$(date +%Y%m%d_%H%M%S)_${EXPERIMENT_NAME}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
    MONITOR_DIR="${OUTPUT_DIR}/monitor"
    LOG_DIR="${OUTPUT_DIR}/logs"
    SNAPSHOT_DIR="${MONITOR_DIR}/code_snapshot"

    mkdir -p "$OUTPUT_DIR" "$MONITOR_DIR" "$LOG_DIR" "$SNAPSHOT_DIR"

    echo "Creating code snapshot..."
    if [ -n "${SNAPSHOT_PATHS:-}" ]; then
        for path in $SNAPSHOT_PATHS; do
            if [ -e "$path" ]; then
                cp -r "$path" "$SNAPSHOT_DIR/"
            fi
        done
    else
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

    if git rev-parse --git-dir > /dev/null 2>&1; then
        git rev-parse HEAD > "$SNAPSHOT_DIR/git_commit.txt" 2>/dev/null || true
        git diff > "$SNAPSHOT_DIR/git_diff.txt" 2>/dev/null || true
        git status > "$SNAPSHOT_DIR/git_status.txt" 2>/dev/null || true
    fi

    cp "$CONFIG_FILE" "$MONITOR_DIR/original_config.conf"
fi

# ----------------------------------------------------------------------------
# Submit Monitor Job
# ----------------------------------------------------------------------------
echo "Submitting monitor job..."

MONITOR_TIME="${MONITOR_DAYS:-7}-00:00:00"

MONITOR_SCRIPT=$(mktemp)
cat > "$MONITOR_SCRIPT" << 'MONITOR_EOF'
#!/bin/bash
#SBATCH --job-name=monitor
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

MONITOR_LOG="$MONITOR_DIR/monitor_detailed.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

log "=========================================="
log "Monitor starting on $(hostname)"
log "Monitor job ID: $SLURM_JOB_ID"
log "Output directory: $OUTPUT_DIR"
log "Monitor directory: $MONITOR_DIR"
log "Log directory: $LOG_DIR"
log "Experiment: $EXPERIMENT_NAME"
log "Max submissions: $MAX_RESUBMISSIONS"
log "Check interval: $CHECK_INTERVAL seconds"
log "Resource type: $RESOURCE_TYPE"
log "Compute partitions: ${COMPUTE_PARTITION_CSV:-None}"
log "Monitor partitions: ${MONITOR_PARTITION_CSV:-None}"
log "Exclude nodes: ${EXCLUDE_NODES:-None}"
log "=========================================="

GPU_JOB_SCRIPT="$MONITOR_DIR/compute_job.sh"

if [ "$RESOURCE_TYPE" = "gpu" ]; then
cat > "$GPU_JOB_SCRIPT" << 'GPU_EOF'
#!/bin/bash
#SBATCH --job-name=compute_task
#SBATCH --gres=gpu:GPU_PLACEHOLDER
#SBATCH --cpus-per-task=CPU_PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER

set -euo pipefail

echo "========================================"
echo "Compute Job Started (GPU)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-None}"
echo "Start time: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

cd "$SNAPSHOT_DIR"
ENV_SETUP
COMMAND_PLACEHOLDER

exit_code=$?

echo "========================================"
echo "Compute Job Finished"
echo "========================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "========================================"

exit $exit_code
GPU_EOF
else
cat > "$GPU_JOB_SCRIPT" << 'CPU_EOF'
#!/bin/bash
#SBATCH --job-name=compute_task
#SBATCH --cpus-per-task=CPU_PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER

set -euo pipefail

echo "========================================"
echo "Compute Job Started (CPU)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

cd "$SNAPSHOT_DIR"
ENV_SETUP
COMMAND_PLACEHOLDER

exit_code=$?

echo "========================================"
echo "Compute Job Finished"
echo "========================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "========================================"

exit $exit_code
CPU_EOF
fi

sed -i "s|GPU_PLACEHOLDER|$GPUS|g" "$GPU_JOB_SCRIPT" 2>/dev/null || true
sed -i "s|CPU_PLACEHOLDER|$CPUS_PER_TASK|g" "$GPU_JOB_SCRIPT"
sed -i "s|MEM_PLACEHOLDER|$MEM|g" "$GPU_JOB_SCRIPT"
sed -i "s|TIME_PLACEHOLDER|$TIME_LIMIT|g" "$GPU_JOB_SCRIPT"

if [ -n "$ENVIRONMENT_SETUP" ]; then
    ENV_SETUP_ESCAPED=$(printf '%s\n' "$ENVIRONMENT_SETUP" | sed -e 's/[\/&]/\\&/g')
    sed -i "s|ENV_SETUP|$ENV_SETUP_ESCAPED|g" "$GPU_JOB_SCRIPT"
else
    sed -i "s|ENV_SETUP|echo 'No environment setup'|g" "$GPU_JOB_SCRIPT"
fi

COMMAND_ESCAPED=$(printf '%s\n' "$COMMAND" | sed -e 's/[\/&]/\\&/g')
sed -i "s|COMMAND_PLACEHOLDER|$COMMAND_ESCAPED|g" "$GPU_JOB_SCRIPT"

chmod +x "$GPU_JOB_SCRIPT"

log "Submitting initial compute job (attempt 1/$MAX_RESUBMISSIONS)..."

SBATCH_ARGS=()
[ -n "$SBATCH_PARTITION_OPT_COMPUTE" ] && SBATCH_ARGS+=("$SBATCH_PARTITION_OPT_COMPUTE")
[ -n "$SBATCH_EXCLUDE_OPT" ] && SBATCH_ARGS+=("$SBATCH_EXCLUDE_OPT")
[ -n "$SBATCH_ACCOUNT_OPT" ] && SBATCH_ARGS+=("$SBATCH_ACCOUNT_OPT")
[ -n "$SBATCH_QOS_OPT" ] && SBATCH_ARGS+=("$SBATCH_QOS_OPT")
[ -n "$SBATCH_CONSTRAINT_OPT" ] && SBATCH_ARGS+=("$SBATCH_CONSTRAINT_OPT")

if ! JOB_ID=$(sbatch \
    "${SBATCH_ARGS[@]}" \
    --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SNAPSHOT_DIR="$SNAPSHOT_DIR" \
    --output="$LOG_DIR/compute_%j.out" \
    --error="$LOG_DIR/compute_%j.err" \
    --parsable \
    "$GPU_JOB_SCRIPT"); then
    log "ERROR: Failed to submit initial compute job"
    touch "$MONITOR_DIR/EXPERIMENT_FAILED"
    exit 1
fi

log "Initial compute job submitted: $JOB_ID"

submission_count=1
echo "compute job $submission_count: $JOB_ID" >> "$MONITOR_DIR/gpu_job_ids.txt"

while true; do
    sleep "$CHECK_INTERVAL"

    if squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; then
        continue
    fi

    log "Compute job $JOB_ID has finished"

    if [ -f "$OUTPUT_DIR/$SUCCESS_FILE" ] || [ -f "$MONITOR_DIR/$SUCCESS_FILE" ]; then
        log "EXPERIMENT COMPLETE (success file: $SUCCESS_FILE)"
        touch "$MONITOR_DIR/EXPERIMENT_COMPLETE"
        exit 0
    fi

    JOB_STATE=$(sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$JOB_STATE" ] && JOB_STATE="UNKNOWN"
    log "Previous job state: $JOB_STATE"

    if [ "$submission_count" -ge "$MAX_RESUBMISSIONS" ]; then
        log "ERROR: Reached maximum submissions ($MAX_RESUBMISSIONS). Last job: $JOB_ID"
        touch "$MONITOR_DIR/EXPERIMENT_FAILED"
        exit 1
    fi

    next_attempt=$((submission_count + 1))
    log "Resubmitting compute job (attempt $next_attempt/$MAX_RESUBMISSIONS)..."

    if ! NEW_JOB_ID=$(sbatch \
        "${SBATCH_ARGS[@]}" \
        --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SNAPSHOT_DIR="$SNAPSHOT_DIR" \
        --output="$LOG_DIR/compute_%j.out" \
        --error="$LOG_DIR/compute_%j.err" \
        --parsable \
        "$GPU_JOB_SCRIPT"); then
        log "ERROR: Failed to resubmit compute job on attempt $next_attempt"
        sleep 300
        continue
    fi

    JOB_ID="$NEW_JOB_ID"
    submission_count=$next_attempt

    log "Compute job resubmitted: $JOB_ID (submission #$submission_count)"
    echo "compute job $submission_count: $JOB_ID" >> "$MONITOR_DIR/gpu_job_ids.txt"
done
MONITOR_EOF

chmod +x "$MONITOR_SCRIPT"

# Build monitor sbatch args
MONITOR_SBATCH_ARGS=()
[ -n "$SBATCH_PARTITION_OPT_MONITOR" ] && MONITOR_SBATCH_ARGS+=("$SBATCH_PARTITION_OPT_MONITOR")
[ -n "$SBATCH_EXCLUDE_OPT" ] && MONITOR_SBATCH_ARGS+=("$SBATCH_EXCLUDE_OPT")
[ -n "$SBATCH_ACCOUNT_OPT" ] && MONITOR_SBATCH_ARGS+=("$SBATCH_ACCOUNT_OPT")
[ -n "$SBATCH_QOS_OPT" ] && MONITOR_SBATCH_ARGS+=("$SBATCH_QOS_OPT")
[ -n "$SBATCH_CONSTRAINT_OPT" ] && MONITOR_SBATCH_ARGS+=("$SBATCH_CONSTRAINT_OPT")

MONITOR_JOB_ID=$(sbatch \
    "${MONITOR_SBATCH_ARGS[@]}" \
    --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",MONITOR_DIR="$MONITOR_DIR",LOG_DIR="$LOG_DIR",SNAPSHOT_DIR="$SNAPSHOT_DIR",EXPERIMENT_NAME="$EXPERIMENT_NAME",MAX_RESUBMISSIONS="$MAX_RESUBMISSIONS",CHECK_INTERVAL="$CHECK_INTERVAL",TIME_LIMIT="$TIME_LIMIT",CPUS_PER_TASK="$CPUS_PER_TASK",MEM="$MEM",GPUS="$GPUS",SUCCESS_FILE="$SUCCESS_FILE",ENVIRONMENT_SETUP="$ENVIRONMENT_SETUP",COMMAND="$COMMAND",RESOURCE_TYPE="$RESOURCE_TYPE",COMPUTE_PARTITION_CSV="$COMPUTE_PARTITION_CSV",MONITOR_PARTITION_CSV="$MONITOR_PARTITION_CSV",SBATCH_PARTITION_OPT_COMPUTE="$SBATCH_PARTITION_OPT_COMPUTE",SBATCH_PARTITION_OPT_MONITOR="$SBATCH_PARTITION_OPT_MONITOR",SBATCH_EXCLUDE_OPT="$SBATCH_EXCLUDE_OPT",SBATCH_ACCOUNT_OPT="$SBATCH_ACCOUNT_OPT",SBATCH_QOS_OPT="$SBATCH_QOS_OPT",SBATCH_CONSTRAINT_OPT="$SBATCH_CONSTRAINT_OPT",INCLUDE_PARTITIONS="$INCLUDE_PARTITIONS",EXCLUDE_PARTITIONS="$EXCLUDE_PARTITIONS",EXCLUDE_NODES="$EXCLUDE_NODES",COMPUTE_PARTITION_MODE="$COMPUTE_PARTITION_MODE",MONITOR_PARTITION_MODE="$MONITOR_PARTITION_MODE",COMPUTE_INCLUDE_PARTITIONS="$COMPUTE_INCLUDE_PARTITIONS",COMPUTE_EXCLUDE_PARTITIONS="$COMPUTE_EXCLUDE_PARTITIONS",MONITOR_INCLUDE_PARTITIONS="$MONITOR_INCLUDE_PARTITIONS",MONITOR_EXCLUDE_PARTITIONS="$MONITOR_EXCLUDE_PARTITIONS" \
    --time="$MONITOR_TIME" \
    --output="$LOG_DIR/monitor_%j.out" \
    --error="$LOG_DIR/monitor_%j.err" \
    --parsable \
    "$MONITOR_SCRIPT")

rm "$MONITOR_SCRIPT"

if [ -z "$MONITOR_JOB_ID" ]; then
    echo "ERROR: Failed to submit monitor job"
    exit 1
fi

echo "Monitor job ID: $MONITOR_JOB_ID" > "$MONITOR_DIR/job_ids.txt"
echo "Started: $(date)" >> "$MONITOR_DIR/job_ids.txt"

echo "========================================"
echo "Experiment Started Successfully!"
echo "========================================"
echo "Monitor job ID: $MONITOR_JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Monitor directory: $MONITOR_DIR"
echo "Log directory: $LOG_DIR"
echo "Code snapshot: $SNAPSHOT_DIR"
echo "Resource type: $RESOURCE_TYPE"
echo "Compute partitions: ${COMPUTE_PARTITION_CSV:-None}"
echo "Monitor partitions: ${MONITOR_PARTITION_CSV:-None}"
echo "Exclude nodes: ${EXCLUDE_NODES:-None}"
echo "Time limit: $TIME_LIMIT"
echo "========================================"