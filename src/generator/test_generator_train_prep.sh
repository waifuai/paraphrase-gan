#!/bin/sh

# Test file for generator train prep

set -x

rm -rf "$(dirname "$0")/generator_train/generator_train_prep/data"
mkdir -p "$(dirname "$0")/generator_train/generator_train_prep/data/input"

# Create mock data for testing
python3 "$(dirname "$0")/generator_train/generator_train_prep/gen_mock_paraphrases.py

# Run the generator train prep script
sh "$(dirname "$0")/generator_train/generator_train_prep/generator_train_prep.sh
