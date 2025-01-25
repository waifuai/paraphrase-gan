#!/bin/sh

# Prepares dataset for training of a discriminator to discriminate whether
# phrases are human-generated or not.

echo "discriminator_train_prep.sh"
stty size | perl -ale 'print "-"x$F[1]'
cat "$(dirname "$0")/disc_train_prep.ascii

set -x

# Clear and create output directory
rm -rf "$(dirname "$0")/data/output"
mkdir -p "$(dirname "$0")/data/output"

# Trax data generation command
PROBLEM_NAME="phrase_discriminator_problem"
DATA_DIR="$(dirname "$0")/data/input"

python -m trax.data.tf_inputs \
    --output_dir="$(dirname "$0")/data/output" \
    --params="$PROBLEM_NAME" \
    --n_shards=1 \
    --input_dir="$DATA_DIR"
