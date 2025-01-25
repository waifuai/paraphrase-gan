#!/bin/sh

# Test file for discriminator train prep
# Creates the sample input files described in the README for discriminator train prep
# and runs the discriminator train prep

set -x

rm -rf   "$(dirname "$0")/discriminator_train/discriminator_train_prep/data
mkdir -p "$(dirname "$0")/discriminator_train/discriminator_train_prep/data/input

python3  "$(dirname "$0")/discriminator_train/gen_mock_paraphrases.py

sh       "$(dirname "$0")/discriminator_train/discriminator_train_prep/discriminator_train_prep.sh
