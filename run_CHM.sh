#!/usr/bin/env bash

# ============================================================
# Used for running full site tests from start to finish
# This file should be stored OUTSIDE the CHM project directory
# ============================================================


# ============================================================
# Configuration
# ============================================================

# Specify site available in sites.json
SITE="fb"

# Directory containing this bash script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_DIR="$SCRIPT_DIR/CHM-MS-net-Canopy-Height-Model"

LOG_DIR="$SCRIPT_DIR/logs"

# ============================================================
# Setup
# ============================================================

# Timestamp for this entire run
RUN_TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"

RUN_DIR="${LOG_DIR}/${SITE}_run_${RUN_TIMESTAMP}"

mkdir -p "$RUN_DIR"

# Overall log
RUN_LOG="$RUN_DIR/run.log"

# Make sure we are using the project directory
cd "$PROJECT_DIR" || {
    echo "ERROR: Could not change to project directory: $PROJECT_DIR"
    exit 1
}

# ============================================================
# Logging helpers
# ============================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RUN_LOG"
}

# ============================================================
# Run a Python script
# ============================================================

# logs are currently disabled (may want to handle case for excluding train.py)
run_python() {
    local script="$1"
    shift

    local script_name
    script_name="$(basename "$script" .py)"

    local log_file="$RUN_DIR/${script_name}.log"

    local start_time
    local end_time
    local elapsed

    log "============================================================"
    log "Starting: python $script $*"
    log "Log file: $log_file"

    start_time=$(date +%s)

    # if ! [[ "$script_name" == "train.py" ]]; then

    #     (
    #         echo "============================================================"
    #         echo "Script: python $script $*"
    #         echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    #         echo "============================================================"
    #         echo

    #         python "$script" "$@"

    #         exit_code=$?

    #         echo
    #         echo "============================================================"
    #         echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    #         echo "Exit code: $exit_code"
    #         echo "============================================================"

    #         exit "$exit_code"
    #     ) > "$log_file" 2>&1
    
    # fi

    (
        echo "============================================================"
        echo "Script: python $script $*"
        echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
        echo

        python "$script" "$@"

        exit_code=$?

        echo
        echo "============================================================"
        echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Exit code: $exit_code"
        echo "============================================================"

        exit "$exit_code"
     ) > "$log_file" 2>&1

    exit_code=$?

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    if [ "$exit_code" -eq 0 ]; then
        log "SUCCESS: $script"
    else
        log "ERROR: $script failed with exit code $exit_code"
    fi

    log "Elapsed time: ${elapsed} seconds"
    log "Finished: $script"

    # Return the Python script's exit code
    return "$exit_code"
}

# ============================================================
# Start run
# ============================================================

log "============================================================"
log "Starting new run"
log "Run timestamp: $RUN_TIMESTAMP"
log "Project directory: $PROJECT_DIR"
log "Python: $(which python)"
log "Python version: $(python --version 2>&1)"
log "============================================================"

FAILED=0

# ============================================================
# Run Python programs
# ============================================================

run_all() {

    # enter prepTrainInputs
    cd prepTrainInputs || return 1

    # run_python "main1.py" --site "$SITE" || return 1

    # run_python "main2.py" --site "$SITE" || return 1

    # # enter ms_net
    # cd ../ms_net

    # run_python "train.py" --site "$SITE" || return 1

    cd ../infer

    run_python "main.py" --site "$SITE" --scale "max" || return 1

    return 0
}

FAILED=0

if ! run_all; then
    FAILED=1
fi

# ============================================================
# Finish
# ============================================================

log "============================================================"

if [ "$FAILED" -eq 0 ]; then
    log "RUN COMPLETED SUCCESSFULLY"
    log "go to vegas immediately."
else
    log "RUN COMPLETED WITH ERRORS"
fi

log "Logs saved in: $RUN_DIR"
log "============================================================"

exit "$FAILED"