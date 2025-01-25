#!/bin/bash
source "${BASE_DIR}/scripts/common.sh"

train_model() {
    local model_type=$1
    local config_file=$2
    
    load_config
    log INFO "Starting ${model_type} training"
    
    # Common training command structure
    python3 -m trax.supervised.trainer \
        --problem="${PROBLEM_NAME}" \
        --model="${MODEL_ARCH}" \
        --data_dir="${DATA_DIR}/processed/${model_type}" \
        --output_dir="${MODELS_DIR}/${model_type}" \
        --config_file="${config_file}" \
        --train_steps="${TRAIN_STEPS}" \
        --eval_frequency="${EVAL_FREQ}"
    
    log SUCCESS "${model_type} training completed"
}