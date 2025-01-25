#!/bin/sh

# Prepares the GAN by training paraphrase generator and phrase discriminator models.
# Improved error handling and resource management
source "${CONFIG_DIR}/paths.sh"
source "${SCRIPTS_DIR}/common.sh"

MAX_RETRIES=3
RETRY_DELAY=5

echo "0_gan_prep.sh"
printf '%*s\n' "$(tput cols)" '' | tr ' ' -
cat "$(dirname "$0")/gan_prep.ascii

set -e  # Exit immediately on error

BASE_DIR="$(dirname "$0")"
DATA_DIR="$BASE_DIR/data/input"
LOCAL_DATA_DIR="$(dirname "$0")/local_data"

# Function to train a model (generator or discriminator)
train_model() {
    local model_dir="$1"
    local train_script="$2"
    
    for ((i=1; i<=MAX_RETRIES; i++)); do
        if bash "${train_script}"; then
            log INFO "Training succeeded for ${model_dir}"
            return 0
        else
            log WARNING "Training failed (attempt ${i}/${MAX_RETRIES})"
            sleep ${RETRY_DELAY}
        fi
    done
    
    log ERROR "Failed to train ${model_dir} after ${MAX_RETRIES} attempts"
    return 1
}

# Train Generator
train_model "$BASE_DIR/../generator/generator_train" "$BASE_DIR/../generator/generator_train/generator_train.sh"

# Train Discriminator
train_model "$BASE_DIR/../discriminator/discriminator_train" "$BASE_DIR/../discriminator/discriminator_train/discriminator_train.sh"

echo "GAN preparation complete."
