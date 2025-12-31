#!/bin/bash
# ============================================================================
# Experiment Management Utilities
# ============================================================================

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/mojtaba.kananisarche/Zh/outputs}"
LOG_ROOT="${LOG_ROOT:-logs}"

# ============================================================================
# Functions
# ============================================================================

list_experiments() {
    echo "========================================"
    echo "Active Experiments"
    echo "========================================"
    echo ""
    
    if [ ! -d "$OUTPUT_ROOT" ]; then
        echo "No experiments found (output directory does not exist)"
        return
    fi
    
    for exp_dir in "$OUTPUT_ROOT"/*/; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        run_id=$(basename "$exp_dir")
        
        if [ -f "$exp_dir/job_ids.txt" ]; then
            monitor_id=$(grep "Monitor job ID:" "$exp_dir/job_ids.txt" | cut -d' ' -f4)
            
            echo "Run ID: $run_id"
            echo "  Monitor Job: $monitor_id"
            
            # Check if job is still running
            if squeue -j "$monitor_id" 2>/dev/null | grep -q "$monitor_id"; then
                echo "  Status: RUNNING"
            else
                if [ -f "$exp_dir/EXPERIMENT_COMPLETE" ] || [ -f "$exp_dir/nested_cv_results.json" ]; then
                    echo "  Status: COMPLETE"
                elif [ -f "$exp_dir/EXPERIMENT_FAILED" ]; then
                    echo "  Status: FAILED"
                else
                    echo "  Status: UNKNOWN/FINISHED"
                fi
            fi
            
            if [ -f "$exp_dir/experiment_config.txt" ]; then
                exp_name=$(grep "Experiment name:" "$exp_dir/experiment_config.txt" | cut -d' ' -f3-)
                echo "  Experiment: $exp_name"
            fi
            
            echo "  Output: $exp_dir"
            echo ""
        fi
    done
}

show_experiment() {
    if [ $# -ne 1 ]; then
        echo "Usage: $0 show <run_id>"
        return 1
    fi
    
    run_id="$1"
    exp_dir="$OUTPUT_ROOT/$run_id"
    
    if [ ! -d "$exp_dir" ]; then
        echo "ERROR: Experiment not found: $run_id"
        return 1
    fi
    
    echo "========================================"
    echo "Experiment: $run_id"
    echo "========================================"
    echo ""
    
    if [ -f "$exp_dir/experiment_config.txt" ]; then
        cat "$exp_dir/experiment_config.txt"
        echo ""
    fi
    
    if [ -f "$exp_dir/job_ids.txt" ]; then
        echo "========================================"
        echo "Job Information"
        echo "========================================"
        cat "$exp_dir/job_ids.txt"
        echo ""
    fi
    
    if [ -f "$exp_dir/gpu_job_ids.txt" ]; then
        echo "========================================"
        echo "GPU Job Submissions"
        echo "========================================"
        cat "$exp_dir/gpu_job_ids.txt"
        echo ""
    fi
    
    if [ -f "$exp_dir/monitor_detailed.log" ]; then
        echo "========================================"
        echo "Recent Monitor Log (last 20 lines)"
        echo "========================================"
        tail -n 20 "$exp_dir/monitor_detailed.log"
        echo ""
    fi
}

cancel_experiment() {
    if [ $# -ne 1 ]; then
        echo "Usage: $0 cancel <run_id>"
        return 1
    fi
    
    run_id="$1"
    exp_dir="$OUTPUT_ROOT/$run_id"
    
    if [ ! -d "$exp_dir" ]; then
        echo "ERROR: Experiment not found: $run_id"
        return 1
    fi
    
    if [ -f "$exp_dir/job_ids.txt" ]; then
        monitor_id=$(grep "Monitor job ID:" "$exp_dir/job_ids.txt" | cut -d' ' -f4 || true)
        
        if [ -n "$monitor_id" ]; then
            echo "Cancelling monitor job: $monitor_id"
            scancel "$monitor_id" 2>/dev/null || echo "  (already finished)"
        fi
    fi
    
    if [ -f "$exp_dir/gpu_job_ids.txt" ]; then
        echo "Cancelling GPU jobs..."
        while read -r line; do
            job_id=$(echo "$line" | awk '{print $NF}')
            if [ -n "$job_id" ] && [[ "$job_id" =~ ^[0-9]+$ ]]; then
                scancel "$job_id" 2>/dev/null || true
            fi
        done < "$exp_dir/gpu_job_ids.txt"
    fi
    
    echo "Experiment cancelled: $run_id"
}

tail_log() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 tail <run_id> [monitor|gpu]"
        return 1
    fi
    
    run_id="$1"
    log_type="${2:-monitor}"
    exp_dir="$OUTPUT_ROOT/$run_id"
    
    if [ ! -d "$exp_dir" ]; then
        echo "ERROR: Experiment not found: $run_id"
        return 1
    fi
    
    if [ "$log_type" = "monitor" ]; then
        if [ -f "$exp_dir/monitor_detailed.log" ]; then
            echo "Tailing monitor log (Ctrl+C to exit)..."
            tail -f "$exp_dir/monitor_detailed.log"
        else
            echo "ERROR: Monitor log not found"
            return 1
        fi
    elif [ "$log_type" = "gpu" ]; then
        log_dir="$LOG_ROOT/$run_id"
        if [ -d "$log_dir" ]; then
            latest_gpu_log=$(ls -t "$log_dir"/gpu_*.out 2>/dev/null | head -1)
            if [ -n "$latest_gpu_log" ]; then
                echo "Tailing latest GPU log (Ctrl+C to exit)..."
                tail -f "$latest_gpu_log"
            else
                echo "ERROR: No GPU logs found"
                return 1
            fi
        else
            echo "ERROR: Log directory not found"
            return 1
        fi
    else
        echo "ERROR: Invalid log type. Use 'monitor' or 'gpu'"
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
    
    run_id="$1"
    exp_dir="$OUTPUT_ROOT/$run_id"
    
    if [ ! -d "$exp_dir" ]; then
        echo "ERROR: Experiment not found: $run_id"
        return 1
    fi
    
    if [ ! -f "$exp_dir/original_config.conf" ]; then
        echo "ERROR: Original config not found. Cannot resume."
        echo "The experiment may have been created with an older version of the system."
        return 1
    fi
    
    if [ ! -d "$exp_dir/code_snapshot" ]; then
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
    if [ -f "$exp_dir/EXPERIMENT_COMPLETE" ] || [ -f "$exp_dir/$SUCCESS_FILE" ]; then
        echo "⚠️  WARNING: This experiment appears to be complete."
        echo ""
    fi
    
    if [ -f "$exp_dir/gpu_job_ids.txt" ]; then
        submission_count=$(wc -l < "$exp_dir/gpu_job_ids.txt")
        echo "Previous GPU job submissions: $submission_count"
    fi
    
    if [ -f "$exp_dir/monitor_detailed.log" ]; then
        echo ""
        echo "Last few monitor log entries:"
        tail -n 5 "$exp_dir/monitor_detailed.log"
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
    SUBMIT_SCRIPT="./submit_job.sh"
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
    "$SUBMIT_SCRIPT" "$exp_dir/original_config.conf" --resume "$run_id"
}

# ============================================================================
# Main
# ============================================================================

case "${1:-}" in
    list)
        list_experiments
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
        echo "  list                    - List all experiments"
        echo "  show <run_id>          - Show details of an experiment"
        echo "  cancel <run_id>        - Cancel an experiment"
        echo "  tail <run_id> [type]   - Tail logs (type: monitor or gpu)"
        echo "  resume <run_id>        - Resume a failed experiment"
        echo ""
        echo "Examples:"
        echo "  $0 list"
        echo "  $0 show 20241229_120000_E14"
        echo "  $0 tail 20241229_120000_E14 monitor"
        echo "  $0 cancel 20241229_120000_E14"
        exit 1
        ;;
esac