#!/bin/bash
source "${BASE_DIR}/scripts/common.sh"

prepare_data() {
    local model_type=$1
    local phase=$2
    
    load_config
    prepare_model_data "${model_type}" "${phase}"
}

# Example usage:
# prepare_data generator train