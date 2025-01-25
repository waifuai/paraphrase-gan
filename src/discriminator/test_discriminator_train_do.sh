#!/bin/sh

# Test file for discriminator train do
# Creates the sample input files described in the README for discriminator train do
# and runs the discriminator train do

set -x

rm -rf   "$(dirname $0)"/discriminator_train/discriminator_train_do/data
rm -rf   "$(dirname $0)"/discriminator_train/discriminator_train_do/t2t_data
mkdir -p "$(dirname $0)"/discriminator_train/discriminator_train_do/data/input

# Create mock data for testing
rm -rf   "$(dirname "$0")/discriminator_train/discriminator_train_do/data
mkdir -p "$(dirname "$0")/discriminator_train/discriminator_train_do/data/input

python3  "$(dirname "$0")/discriminator_train/gen_mock_paraphrases.py

sh       "$(dirname $0)"/discriminator_train/discriminator_train_do/discriminator_train_do.sh
