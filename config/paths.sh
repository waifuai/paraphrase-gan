#!/bin/bash
# Consistent naming convention and added logging
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Directory structure
export CONFIG_DIR="${PROJECT_ROOT}/config"
export SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
export SRC_DIR="${PROJECT_ROOT}/src"
export DATA_ROOT="${PROJECT_ROOT}/data"
export MODELS_ROOT="${PROJECT_ROOT}/models"
export LOGS_DIR="${PROJECT_ROOT}/logs"

# Subdirectories
export RAW_DATA_DIR="${DATA_ROOT}/raw"
export PROCESSED_DATA_DIR="${DATA_ROOT}/processed"
export GENERATOR_MODELS="${MODELS_ROOT}/generator"
export DISCRIMINATOR_MODELS="${MODELS_ROOT}/discriminator"

# Create directories if missing
mkdir -p "${LOGS_DIR}" "${RAW_DATA_DIR}" "${PROCESSED_DATA_DIR}" \
         "${GENERATOR_MODELS}" "${DISCRIMINATOR_MODELS}"