#!/bin/sh

# Tests the GAN prep script by creating sample input and running the script.

set -x

# Create mock paraphrase files for testing.
mkdir -p "$(dirname "$0")/data/input"
python3 "$(dirname "$0")/gen_mock_paraphrases.py

# Run the GAN prep script.
sh "$(dirname "$0")/0_gan_prep.sh
