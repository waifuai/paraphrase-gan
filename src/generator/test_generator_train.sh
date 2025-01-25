#!/bin/sh

# Test file for generator train do
# Creates the sample input files described in the README for generator train do
# and runs the generator train do

set -x

rm -rf   "$(dirname $0)"/generator_train/data
rm -rf   "$(dirname $0)"/generator_train/t2t_data
mkdir -p "$(dirname $0)"/generator_train/data/input

rm -rf   "$(dirname "$0")/generator_train/data
rm -rf   "$(dirname "$0")/generator_train/t2t_data
mkdir -p "$(dirname "$0")/generator_train/data/input

# Create mock data for testing
python3  "$(dirname "$0")/generator_train/generator_train_prep/gen_mock_paraphrases.py

# Run the generator train script
sh "$(dirname "$0")/generator_train/generator_train_do/generator_train_do.sh
