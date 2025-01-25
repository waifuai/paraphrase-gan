#!/bin/sh

# Trains a paraphrase generator.
# Replaces the previous generator model with a newly trained generator model.

echo "generator_train_do.sh"
stty size | perl -ale 'print "-"x$F[1]'
cat "$(dirname "$0")/gen_train_do.ascii

set -x

# Define model and data paths
MODEL_NAME="transformer_phrase_generator"
PROBLEM_NAME="phrase_generator_problem"
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

echo "Generator training complete. Model saved to $MODEL_DIR"
