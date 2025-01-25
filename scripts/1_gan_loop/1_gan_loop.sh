#!/bin/sh

# Runs a single iteration of the GAN loop

echo "1_gan_loop.sh"
cat "$(dirname "$0")/gan_loop.ascii

set -e  # Exit immediately on error

BASE_DIR="$(dirname "$0")"
DATA_DIR="$BASE_DIR/data/input"
LOCAL_DATA_DIR="$BASE_DIR/local_data"
GENERATOR_DIR="$BASE_DIR/../generator/generator_generate"
DISCRIMINATOR_DIR="$BASE_DIR/../discriminator/discriminator_discriminate"
GENERATOR_TRAIN_DIR="$BASE_DIR/../generator/generator_train"

# Function to prepare data for a model (generator or discriminator)
prepare_data() {
    local model_dir="$1"
    local input_dir="$2"

    rm -rf "$model_dir/data"
    mkdir -p "$model_dir/data/input"
    cp -r "$input_dir" "$model_dir/data"
}

# Generate Phrases
prepare_data "$GENERATOR_DIR" "$DATA_DIR"
sh "$GENERATOR_DIR/generator_generate.sh"

# Discriminate Phrases
prepare_data "$DISCRIMINATOR_DIR" "$GENERATOR_DIR/data/output"
sh "$DISCRIMINATOR_DIR/discriminator_discriminate.sh"

# Add accepted paraphrases to generator training data
prepare_data "$GENERATOR_TRAIN_DIR" "$DATA_DIR"

# Copy the accepted paraphrases from discriminator output to generator training input
cp "$DISCRIMINATOR_DIR/data/output/paraphrases_selected.tsv" "$GENERATOR_TRAIN_DIR/data/input/paraphrases.tsv"

# Train Generator
sh "$GENERATOR_TRAIN_DIR/generator_train.sh"

echo "GAN loop iteration complete."
