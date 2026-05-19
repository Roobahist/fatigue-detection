#!/bin/bash
# ============================================================================
# Experiment Management Utilities
# ============================================================================

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/mojtaba.kananisarche/Zh/outputs}"

# ============================================================================
# Helper Functions
# ============================================================================

get_job_status() {
    local job_id="$1"

    # First check if job is in queue
    if squeue -j "$job_id" &>/dev/null; then
        local state
        state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null | head -1)
        if [ -n "$state" ]; then
            echo "$state"
            return 0
        fi
    fi

    # If not in queue, check accounting (recent history)
    local state
    state=$(sacct -j "$job_id" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$state" ]; then
        echo "$state"
        return 0
    fi

    echo "NOT_FOUND"
    return 1
}

is_job_running() {
    local job_id="$1"
    local status
    status=$(get_job_status "$job_id")

    case "$status" in
        RUNNING|PENDING|CONFIGURING|COMPLETING|RESIZING|SUSPENDED)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

get_current_gpu_job() {
    local monitor_dir="$1"

    if [ ! -f "$monitor_dir/gpu_job_ids.txt" ]; then
        echo ""
        return 1
    fi

    # Get the most recent GPU job ID
    local latest_job
    latest_job=$(tail -1 "$monitor_dir/gpu_job_ids.txt" | awk '{print $NF}')

    if [ -n "$latest_job" ] && [[ "$latest_job" =~ ^[0-9]+$ ]]; then
        echo "$latest_job"
        return 0
    fi

    echo ""
    return 1
}

# ============================================================================
# Main Functions
# ============================================================================

list_experiments() {
    local limit="${1:-0}"  # 0 means no limit
    local count=0

    echo "========================================"
    echo "Experiments"
    echo "========================================"
    echo ""

    if [ ! -d "$OUTPUT_ROOT" ]; then
        echo "No experiments found (output directory does not exist: $OUTPUT_ROOT)"
        return
    fi

    local found_any=false

    # Build a list of experiment dirs that have a monitor/ dir, sorted by most recently modified
    # Format: "<mtime_epoch> <path>"
    local exp_lines
    exp_lines=$(
        find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null \
        | while IFS= read -r -d '' d; do
            if [ -d "$d/monitor" ]; then
                printf '%s %s\n' "$(stat -c %Y "$d" 2>/dev/null || echo 0)" "$d"
            fi
        done \
        | sort -nr
    )

    if [ -z "$exp_lines" ]; then
        echo "No experiments found in $OUTPUT_ROOT"
        return
    fi

    while IFS= read -r line; do
        [ -z "$line" ] && continue

        # Respect limit if provided
        if [ "$limit" -gt 0 ] && [ "$count" -ge "$limit" ]; then
            break
        fi

        # line: "<mtime> <dir>"
        local exp_dir
        exp_dir=$(echo "$line" | cut -d' ' -f2-)
        [ ! -d "$exp_dir" ] && continue

        local run_id
        run_id=$(basename "$exp_dir")
        local monitor_dir="$exp_dir/monitor"

        found_any=true
        count=$((count + 1))

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Run ID: $run_id"

        # Get experiment name
        if [ -f "$monitor_dir/experiment_config.txt" ]; then
            local exp_name
            exp_name=$(grep "Experiment name:" "$monitor_dir/experiment_config.txt" | cut -d':' -f2- | sed 's/^ *//' || true)
            if [ -n "$exp_name" ]; then
                echo "  Name: $exp_name"
            fi

            # Optional: show partitions if present in experiment_config.txt
            local parts
            parts=$(grep "^Partitions:" "$monitor_dir/experiment_config.txt" | head -1 | cut -d':' -f2- | sed 's/^ *//' || true)
            if [ -n "$parts" ]; then
                echo "  Partitions: $parts"
            fi

            # Optional: show excluded nodes if present
            local excl
            excl=$(grep "^Exclude nodes:" "$monitor_dir/experiment_config.txt" | head -1 | cut -d':' -f2- | sed 's/^ *//' || true)
            if [ -n "$excl" ]; then
                echo "  Exclude nodes: $excl"
            fi
        fi

        # Check monitor job status
        local monitor_status="UNKNOWN"
        local monitor_id=""

        if [ -f "$monitor_dir/job_ids.txt" ]; then
            monitor_id=$(grep "Monitor job ID:" "$monitor_dir/job_ids.txt" | awk '{print $NF}' || true)

            if [ -n "$monitor_id" ]; then
                monitor_status=$(get_job_status "$monitor_id")
                echo "  Monitor Job: $monitor_id"
                echo "    Status: $monitor_status"
            fi
        fi

        # Check GPU job status
        local gpu_job_id
        gpu_job_id=$(get_current_gpu_job "$monitor_dir")
        if [ -n "$gpu_job_id" ]; then
            local gpu_status
            gpu_status=$(get_job_status "$gpu_job_id")
            echo "  Current GPU Job: $gpu_job_id"
            echo "    Status: $gpu_status"
        fi

        # Count total GPU submissions
        if [ -f "$monitor_dir/gpu_job_ids.txt" ]; then
            local submission_count
            submission_count=$(wc -l < "$monitor_dir/gpu_job_ids.txt")
            echo "  GPU Submissions: $submission_count"
        fi

        # Overall experiment status
        local overall_status=""

        if [ -n "$monitor_id" ] && is_job_running "$monitor_id" 2>/dev/null; then
            overall_status="🟢 RUNNING"
        elif [ -f "$monitor_dir/EXPERIMENT_COMPLETE" ] || [ -f "$exp_dir/EXPERIMENT_COMPLETE" ]; then
            overall_status="✅ COMPLETE"
        elif [ -f "$monitor_dir/EXPERIMENT_FAILED" ]; then
            overall_status="❌ FAILED"
        else
            # Monitor finished but no complete/failed marker
            overall_status="⚠️  STOPPED"
        fi

        echo "  Overall Status: $overall_status"

        # Show start time
        if [ -f "$monitor_dir/job_ids.txt" ]; then
            local start_time
            start_time=$(grep "Started:" "$monitor_dir/job_ids.txt" | cut -d':' -f2- | sed 's/^ *//' || true)
            if [ -n "$start_time" ]; then
                echo "  Started: $start_time"
            fi
        fi

        echo "  Path: $exp_dir"
        echo ""
    done <<< "$exp_lines"

    if [ "$found_any" = false ]; then
        echo "No experiments found in $OUTPUT_ROOT"
    fi
}

show_experiment() {
    if [ $# -ne 1 ]; then
        echo "Usage: $0 show <run_id>"
        return 1
    fi

    local run_id="$1"
    local exp_dir="$OUTPUT_ROOT/$run_id"
    local monitor_dir="$exp_dir/monitor"

    if [ ! -d "$exp_dir" ]; then
        echo "ERROR: Experiment not found: $run_id"
        return 1
    fi

    if [ ! -d "$monitor_dir" ]; then
        echo "ERROR: Monitor directory not found: $monitor_dir"
        echo "This may be an old experiment with different structure."
        return 1
    fi

    echo "========================================"
    echo "Experiment: $run_id"
    echo "========================================"
    echo ""

    # Configuration
    if [ -f "$monitor_dir/experiment_config.txt" ]; then
        echo "━━━ Configuration ━━━"
        cat "$monitor_dir/experiment_config.txt"
        echo ""
    fi

    # Job Information
    echo "━━━ Job Status ━━━"

    local monitor_id=""
    if [ -f "$monitor_dir/job_ids.txt" ]; then
        monitor_id=$(grep "Monitor job ID:" "$monitor_dir/job_ids.txt" | awk '{print $NF}' || true)

        if [ -n "$monitor_id" ]; then
            local monitor_status
            monitor_status=$(get_job_status "$monitor_id")
            echo "Monitor Job: $monitor_id"
            echo "  Status: $monitor_status"

            # Show detailed squeue info if running
            if is_job_running "$monitor_id"; then
                echo "  Details:"
                squeue -j "$monitor_id" -o "    Time: %M | Node: %N | State: %T" 2>/dev/null || true
            fi
        fi
    fi

    # Current GPU job
    local gpu_job_id
    gpu_job_id=$(get_current_gpu_job "$monitor_dir")
    if [ -n "$gpu_job_id" ]; then
        local gpu_status
        gpu_status=$(get_job_status "$gpu_job_id")
        echo ""
        echo "Current GPU Job: $gpu_job_id"
        echo "  Status: $gpu_status"

        if is_job_running "$gpu_job_id"; then
            echo "  Details:"
            squeue -j "$gpu_job_id" -o "    Time: %M | Node: %N | State: %T | GPU: %b" 2>/dev/null || true
        fi
    fi

    echo ""

    # GPU Job History
    if [ -f "$monitor_dir/gpu_job_ids.txt" ]; then
        echo "━━━ GPU Job History ━━━"
        local total
        total=$(wc -l < "$monitor_dir/gpu_job_ids.txt")
        echo "Total submissions: $total"
        echo ""
        echo "Last 5 submissions:"
        tail -n 5 "$monitor_dir/gpu_job_ids.txt" | while read -r line; do
            local job_id
            job_id=$(echo "$line" | awk '{print $NF}')
            local submission_num
            submission_num=$(echo "$line" | awk '{print $3}' | tr -d ':')
            if [ -n "$job_id" ]; then
                local status
                status=$(get_job_status "$job_id")
                echo "  #$submission_num: $job_id - $status"
            fi
        done
        echo ""
    fi

    # Recent Monitor Log
    if [ -f "$monitor_dir/monitor_detailed.log" ]; then
        echo "━━━ Recent Monitor Log (last 15 lines) ━━━"
        tail -n 15 "$monitor_dir/monitor_detailed.log"
        echo ""
    fi

    # Overall Status
    echo "━━━ Overall Status ━━━"
    if [ -f "$monitor_dir/EXPERIMENT_COMPLETE" ] || [ -f "$exp_dir/EXPERIMENT_COMPLETE" ]; then
        echo "✅ EXPERIMENT COMPLETE"
    elif [ -f "$monitor_dir/EXPERIMENT_FAILED" ]; then
        echo "❌ EXPERIMENT FAILED"
    elif [ -n "$monitor_id" ] && is_job_running "$monitor_id"; then
        echo "🟢 EXPERIMENT RUNNING"
    else
        echo "⚠️  EXPERIMENT STOPPED"
    fi

    echo ""
    echo "Directories:"
    echo "  Output: $exp_dir"
    echo "  Monitor: $monitor_dir"
    echo "  Logs: $exp_dir/logs"
    echo "  Code Snapshot: $monitor_dir/code_snapshot"
}

cancel_experiment() {
    if [ $# -ne 1 ]; then
        echo "Usage: $0 cancel <run_id>"
        return 1
    fi

    local run_id="$1"
    local exp_dir="$OUTPUT_ROOT/$run_id"
    local monitor_dir="$exp_dir/monitor"

    if [ ! -d "$exp_dir" ]; then
        echo "ERROR: Experiment not found: $run_id"
        return 1
    fi

    if [ ! -d "$monitor_dir" ]; then
        echo "ERROR: Monitor directory not found: $monitor_dir"
        return 1
    fi

    echo "Cancelling experiment: $run_id"
    echo ""

    local cancelled_any=false

    # Cancel monitor job
    if [ -f "$monitor_dir/job_ids.txt" ]; then
        local monitor_id
        monitor_id=$(grep "Monitor job ID:" "$monitor_dir/job_ids.txt" | awk '{print $NF}' || true)

        if [ -n "$monitor_id" ]; then
            echo "Checking monitor job $monitor_id..."
            if is_job_running "$monitor_id"; then
                echo "  Cancelling monitor job: $monitor_id"
                scancel "$monitor_id" 2>/dev/null && cancelled_any=true
            else
                echo "  Monitor job already finished"
            fi
        fi
    fi

    # Cancel current GPU job
    local gpu_job_id
    gpu_job_id=$(get_current_gpu_job "$monitor_dir")
    if [ -n "$gpu_job_id" ]; then
        echo "Checking GPU job $gpu_job_id..."
        if is_job_running "$gpu_job_id"; then
            echo "  Cancelling GPU job: $gpu_job_id"
            scancel "$gpu_job_id" 2>/dev/null && cancelled_any=true
        else
            echo "  GPU job already finished"
        fi
    fi

    if [ "$cancelled_any" = true ]; then
        echo ""
        echo "✓ Jobs cancelled successfully"
    else
        echo ""
        echo "No running jobs found for this experiment"
    fi
}

tail_log() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 tail <run_id> [monitor|gpu]"
        return 1
    fi

    local run_id="$1"
    local log_type="${2:-monitor}"
    local exp_dir="$OUTPUT_ROOT/$run_id"
    local monitor_dir="$exp_dir/monitor"
    local log_dir="$exp_dir/logs"

    if [ ! -d "$exp_dir" ]; then
        echo "ERROR: Experiment not found: $run_id"
        return 1
    fi

    if [ "$log_type" = "monitor" ]; then
        if [ -f "$monitor_dir/monitor_detailed.log" ]; then
            echo "Tailing monitor log (Ctrl+C to exit)..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -f "$monitor_dir/monitor_detailed.log"
        else
            echo "ERROR: Monitor log not found at $monitor_dir/monitor_detailed.log"
            return 1
        fi
    elif [ "$log_type" = "gpu" ]; then
        if [ -d "$log_dir" ]; then
            local latest_gpu_log
            latest_gpu_log=$(ls -t "$log_dir"/gpu_*.out 2>/dev/null | head -1)
            if [ -n "$latest_gpu_log" ]; then
                echo "Tailing latest GPU log: $(basename "$latest_gpu_log")"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                tail -f "$latest_gpu_log"
            else
                echo "ERROR: No GPU logs found in $log_dir"
                return 1
            fi
        else
            echo "ERROR: Log directory not found: $log_dir"
            return 1
        fi
    else
        echo "ERROR: Invalid log type '$log_type'. Use 'monitor' or 'gpu'"
        return 1
    fi
}

resume_experiment() {
    if [ $# -ne 1 ]; then
        echo "Usage: $0 resume <run_id>"
        echo ""
        echo "This will resubmit the monitor job for a failed/interrupted experiment."
        echo "The experiment will continue from where it left off using the same code snapshot."
        return 1
    fi

    local run_id="$1"
    local exp_dir="$OUTPUT_ROOT/$run_id"
    local monitor_dir="$exp_dir/monitor"

    if [ ! -d "$exp_dir" ]; then
        echo "ERROR: Experiment not found: $run_id"
        return 1
    fi

    if [ ! -d "$monitor_dir" ]; then
        echo "ERROR: Monitor directory not found: $monitor_dir"
        return 1
    fi

    if [ ! -f "$monitor_dir/original_config.conf" ]; then
        echo "ERROR: Original config not found. Cannot resume."
        echo "The experiment may have been created with an older version of the system."
        return 1
    fi

    if [ ! -d "$monitor_dir/code_snapshot" ]; then
        echo "ERROR: Code snapshot not found. Cannot resume."
        return 1
    fi

    echo "========================================"
    echo "Resume Experiment"
    echo "========================================"
    echo "Run ID: $run_id"
    echo "Output: $exp_dir"
    echo ""

    # Show current status
    if [ -f "$monitor_dir/EXPERIMENT_COMPLETE" ] || [ -f "$exp_dir/EXPERIMENT_COMPLETE" ]; then
        echo "⚠️  WARNING: This experiment appears to be complete."
        echo ""
    fi

    if [ -f "$monitor_dir/gpu_job_ids.txt" ]; then
        local submission_count
        submission_count=$(wc -l < "$monitor_dir/gpu_job_ids.txt")
        echo "Previous GPU job submissions: $submission_count"
    fi

    # Check if monitor is still running
    if [ -f "$monitor_dir/job_ids.txt" ]; then
        local monitor_id
        monitor_id=$(grep "Monitor job ID:" "$monitor_dir/job_ids.txt" | awk '{print $NF}' || true)
        if [ -n "$monitor_id" ] && is_job_running "$monitor_id"; then
            echo ""
            echo "⚠️  WARNING: Monitor job $monitor_id is still running!"
            echo "You should cancel it first: ./manage_experiments.sh cancel $run_id"
            return 1
        fi
    fi

    if [ -f "$monitor_dir/monitor_detailed.log" ]; then
        echo ""
        echo "Last few monitor log entries:"
        tail -n 5 "$monitor_dir/monitor_detailed.log"
    fi

    echo ""
    echo "========================================"
    echo "This will:"
    echo "  1. Cancel any running jobs for this experiment"
    echo "  2. Resubmit a new monitor job"
    echo "  3. Continue using the same code snapshot"
    echo "  4. Resume GPU job submissions"
    echo "========================================"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        return 0
    fi

    # Find the submit_job.sh script
    local SUBMIT_SCRIPT="./submit_job.sh"
    if [ ! -f "$SUBMIT_SCRIPT" ]; then
        SUBMIT_SCRIPT="$(dirname "$0")/submit_job.sh"
    fi

    if [ ! -f "$SUBMIT_SCRIPT" ]; then
        echo "ERROR: Cannot find submit_job.sh script"
        echo "Please run this from the directory containing submit_job.sh"
        return 1
    fi

    # Resume using the submit script
    echo ""
    echo "Resuming experiment..."
    "$SUBMIT_SCRIPT" "$monitor_dir/original_config.conf" --resume "$run_id"
}

# ============================================================================
# Main
# ============================================================================

case "${1:-}" in
    list)
        list_experiments "${2:-}"
        ;;
    show)
        shift
        show_experiment "$@"
        ;;
    cancel)
        shift
        cancel_experiment "$@"
        ;;
    tail)
        shift
        tail_log "$@"
        ;;
    resume)
        shift
        resume_experiment "$@"
        ;;
    *)
        echo "Usage: $0 {list|show|cancel|tail|resume} [args]"
        echo ""
        echo "Commands:"
        echo "  list                    - List all experiments with their status"
        echo "  show <run_id>          - Show detailed information about an experiment"
        echo "  cancel <run_id>        - Cancel all jobs for an experiment"
        echo "  tail <run_id> [type]   - Follow logs (type: monitor or gpu, default: monitor)"
        echo "  resume <run_id>        - Resume a stopped/failed experiment"
        echo ""
        echo "Examples:"
        echo "  $0 list"
        echo "  $0 show 20241229_120000_E14"
        echo "  $0 tail 20241229_120000_E14 monitor"
        echo "  $0 cancel 20241229_120000_E14"
        echo "  $0 resume 20241229_120000_E14"
        echo ""
        echo "Environment variables:"
        echo "  OUTPUT_ROOT - Root directory for experiments (default: /home/mojtaba.kananisarche/Zh/outputs)"
        exit 1
        ;;
esac