#!/bin/sh

# Handles preparation and performing the discrimination of whether a given
# phrase is human-generated or not.

echo "discriminator_discriminate.sh"
stty size | perl -ale 'print "-"x$F[1]'
cat "$(dirname "$0")/discriminating.ascii

set -x

# Prepare input data
mkdir -p "$(dirname "$0")/data/input"
cut -f1 "$(dirname "$0")/data/input/paraphrases_generated.tsv > "$(dirname "$0")/data/input/phrases_generated.tsv

# Define model and data paths
MODEL_NAME="transformer_phrase_discriminator"
PROBLEM_NAME="phrase_discriminator_problem"
DATA_DIR="$(dirname "$0")/data/input"
OUTPUT_DIR="$(dirname "$0")/data/output"
MODEL_DIR="$(dirname "$0")/model"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Trax decoder command
python -m trax.supervised.decode \
    --output_dir="$OUTPUT_DIR" \
    --model="$MODEL_NAME" \
    --problem="$PROBLEM_NAME" \
    --data_dir="$DATA_DIR" \
    --decode_steps="100" \
    --decode_in_memory=True

# Rename decode output file
mv "$OUTPUT_DIR/decode_out.txt" "$OUTPUT_DIR/phrases_discrimination_labels.tsv"

python3 "$(dirname "$0")/discriminator_discriminate_postprocess.py
