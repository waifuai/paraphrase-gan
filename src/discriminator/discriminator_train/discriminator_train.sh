#!/bin/sh

# Handles preparation and training a disciminator model to discriminate of
# whether a give phrase is human-generated or not.

echo "discriminator_train.sh"
cat "$(dirname "$0")/disc_train.ascii
stty size | perl -ale 'print "-"x$F[1]'

set -x

# Prepare data for training
mkdir -p "$(dirname "$0")/discriminator_train_prep/data/input"
cp -r "$(dirname "$0")/data/input" "$(dirname "$0")/discriminator_train_prep/data/input"

# Prepare training data
sh "$(dirname "$0")/discriminator_train_prep/discriminator_train_prep.sh"

# Copy prepared data to training directory
mkdir -p "$(dirname "$0")/discriminator_train_do/data/input"
cp -r "$(dirname "$0")/discriminator_train_prep/data/output/*" "$(dirname "$0")/discriminator_train_do/data/input"

# Train the discriminator
sh "$(dirname "$0")/discriminator_train_do/discriminator_train_do.sh"
