#!/bin/bash

# Runs the GAN. This is the file you should run.
# Runs the preparation of GAN once, then runs the GAN training loop iteratively forever

echo "run.sh"
stty size | perl -ale 'print "-"x$F[1]'

# Prepare the GAN (train initial models)
bash "$(dirname "$0")/scripts/0_gan_prep/0_gan_prep.sh"

# Run the GAN loop indefinitely
while true; do
  bash "$(dirname "$0")/scripts/1_gan_loop/1_gan_loop.sh"
done