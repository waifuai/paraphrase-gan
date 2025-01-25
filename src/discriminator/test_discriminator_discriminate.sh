#!/bin/sh

# Test file for discriminator discriminate
# Creates the sample input files described in the README for discriminator discriminate
# and runs the discriminator discriminate

set -x

rm -rf "$(dirname "$0")/discriminator_discriminate/data"
mkdir -p "$(dirname "$0")/discriminator_discriminate/data/input"
mkdir -p "$(dirname "$0")/discriminator_discriminate/data/output"

# Create a mock input file
cat << EOF > "$(dirname "$0")/discriminator_discriminate/data/input/paraphrases_generated.tsv"
a phrase\tanother paraphrase
a phrase\tanother paraphrase of same paraphrase
again a phrase\tagain another paraphrase
EOF

# Run the discriminator
sh "$(dirname "$0")/discriminator_discriminate/discriminator_discriminate.sh
