#!/bin/bash
# Common configuration and functions
# Unified error handling and logging
set -Eeuo pipefail
trap 'handle_error $LINENO' ERR

handle_error() {
    local line=$1
    log ERROR "Error occurred on line $line"
    exit 1
}

log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOGS_DIR}/run.log"
}

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

load_config() {
    source "${BASE_DIR}/config/paths.sh"
}

prepare_model_data() {
    local model_type=$1
    local phase=$2
    log INFO "Preparing ${model_type} data for ${phase}"
    
    # Common data preparation logic
    python3 "${BASE_DIR}/src/utils/prepare_data.py" \
        --model-type "${model_type}" \
        --phase "${phase}" \
        --input-dir "${DATA_DIR}/raw" \
        --output-dir "${DATA_DIR}/processed/${model_type}"
}