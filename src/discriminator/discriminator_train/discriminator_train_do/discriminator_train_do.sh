#!/bin/sh

# Trains a discriminator to discriminate whether phrases are human-generated or not.
# Replaces the previous discriminator model with a newly trained discriminator model.

echo "discriminator_train_do.sh"
stty size | perl -ale 'print "-"x$F[1]'
cat "$(dirname "$0")/disc_train_do.ascii

set -x

# Define model and data paths
MODEL_NAME="transformer_phrase_discriminator"
PROBLEM_NAME="phrase_discriminator_problem"
DATA_DIR="$(dirname "$0")/data/input"
OUTPUT_DIR="$(dirname "$0")/data/output"
MODEL_DIR="$(dirname "$0")/model"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODEL_DIR"

# Trax trainer command
python -m trax.supervised.trainer \
    --output_dir="$OUTPUT_DIR" \
    --model="$MODEL_NAME" \
    --problem="$PROBLEM_NAME" \
    --data_dir="$DATA_DIR" \
    --train_steps="1000" \
    --eval_steps="100" \
    --eval_frequency="100"

echo "Discriminator training complete. Model saved to $MODEL_DIR"
