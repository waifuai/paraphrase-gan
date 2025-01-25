#!/bin/sh

# Prepares data and generates paraphrases.
# Given a set of human phrases, the generator generates machine paraphrases
# of each human phrase it receives.

echo "generator_generate.sh"
stty size | perl -ale 'print "-"x$F[1]'
cat "$(dirname "$0")/generating.ascii

set -x

# Define model and data paths (local)
MODEL_NAME="transformer_phrase_generator"
PROBLEM_NAME="phrase_generator_problem"
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
