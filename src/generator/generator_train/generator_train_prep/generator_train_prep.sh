#!/bin/sh

# Prepares dataset for training the generator

echo "generator_train_prep.sh"
stty size | perl -ale 'print "-"x$F[1]'
cat "$(dirname "$0")/gen_train_prep.ascii

set -x

# Clear and create output directory
rm -rf "$(dirname "$0")/data/output"
mkdir -p "$(dirname "$0")/data/output"

# Trax data generation command
PROBLEM_NAME="phrase_generator_problem"
DATA_DIR="$(dirname "$0")/data/input"

python -m trax.data.tf_inputs \
    --output_dir="$(dirname "$0")/data/output" \
    --params="$PROBLEM_NAME" \
    --n_shards=1 \
    --input_dir="$DATA_DIR"
