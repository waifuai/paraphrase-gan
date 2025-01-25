#!/bin/sh

echo "generator_train.sh"
stty size | perl -ale 'print "-"x$F[1]'
cat "$(dirname "$0")/gen_train.ascii

set -x

# Prepare data for training
mkdir -p "$(dirname "$0")/generator_train_prep/data/input"
cp -r "$(dirname "$0")/data/input" "$(dirname "$0")/generator_train_prep/data/input"

# Prepare training data
sh "$(dirname "$0")/generator_train_prep/generator_train_prep.sh"

# Copy prepared data to training directory
mkdir -p "$(dirname "$0")/generator_train_do/data/input"
cp -r "$(dirname "$0")/generator_train_prep/data/output/*" "$(dirname "$0")/generator_train_do/data/input"

# Train the generator
sh "$(dirname "$0")/generator_train_do/generator_train_do.sh"
