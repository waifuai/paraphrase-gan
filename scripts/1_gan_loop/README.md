# GAN Loop (1_gan_loop)

This directory contains scripts and resources for running a single iteration of the Generative Adversarial Network (GAN) loop. This loop iteratively improves the quality of generated paraphrases by leveraging the interplay between a paraphrase generator and a phrase discriminator.

## Overview

The `1_gan_loop.sh` script implements a single iteration of the GAN loop. Each iteration involves the following steps:

1. **Phrase Generation:** The generator model takes input phrases (from `data/input/paraphrases.tsv`) and generates paraphrases.
2. **Phrase Discrimination:** The discriminator model evaluates the generated paraphrases and attempts to classify them as human-generated or machine-generated. The paraphrases classified as human-generated are considered "accepted."
3. **Generator Training:** The generator is trained further using the "accepted" paraphrases as training data, encouraging it to produce more human-like paraphrases in the future.
4. **Data Preparation:** The training process begins by preparing the data. It copies input data and makes it available for the generator and discriminator to generate and discriminate phrases. The accepted phrases are then put into the correct directory and format, so that the generator can be trained on it.

## Process Details

The `1_gan_loop.sh` script performs the following actions:

1. **Initialization:**
    *   Prints an informative message and displays the GAN loop ASCII art.
    *   Sets up necessary variables, including paths to data directories and model directories.

2. **Data Preparation (for Generator):**
    *   The `prepare_data` function is called to prepare the data for the generator.
    *   It removes any existing data in the generator's data directory and copies the input data (`$DATA_DIR`) to the generator's input directory (`$GENERATOR_DIR/data/input`).

3. **Generate Phrases:**
    *   Executes the generator's phrase generation script (`$GENERATOR_DIR/generator_generate.sh`).
    *   The generator uses its model to generate paraphrases, saving the output in its output directory (`$GENERATOR_DIR/data/output`).

4. **Data Preparation (for Discriminator):**
    *   The `prepare_data` function is called to prepare the data for the discriminator.
    *   It removes existing data and copies the generator's output to the discriminator's input directory (`$DISCRIMINATOR_DIR/data/input`).

5. **Discriminate Phrases:**
    *   Executes the discriminator's phrase discrimination script (`$DISCRIMINATOR_DIR/discriminator_discriminate.sh`).
    *   The discriminator analyzes the generated paraphrases and selects those it believes are human-generated, storing them in its output directory (`$DISCRIMINATOR_DIR/data/output`). The output will be in the discriminator's output directory under the name `paraphrases_selected.tsv`.

6. **Data Preparation (for Generator Training):**
    *   The `prepare_data` function is called again to set up the data for training the generator.
    *   The initial input data is copied to the generator training input directory (`$GENERATOR_TRAIN_DIR/data/input`).

7. **Copy Accepted Paraphrases:**
    *   The accepted paraphrases (`paraphrases_selected.tsv`) from the discriminator's output are copied to the generator's training input directory (`$GENERATOR_TRAIN_DIR/data/input`), overwriting the file `paraphrases.tsv`. This effectively adds the newly accepted paraphrases to the generator's training dataset.

8. **Train Generator:**
    *   Executes the generator's training script (`$GENERATOR_TRAIN_DIR/generator_train.sh`).
    *   The generator is trained on the updated dataset, including the newly accepted paraphrases.

9. **Completion:**
    *   Prints a message indicating the completion of a single GAN loop iteration.

## Data

### Input

*   **`scripts/1_gan_loop/data/input/paraphrases.tsv`:**  A tab-separated file containing input phrases for the generator. Each line contains a phrase and a paraphrase, formatted as `phrase\tparaphrase`.

### Intermediate Data

*   **`scripts/generator/generator_generate/data/output/`:** Output directory of the generator after generating paraphrases.
*   **`scripts/discriminator/discriminator_discriminate/data/output/`:** Output directory of the discriminator after discriminating phrases. Contains `paraphrases_selected.tsv` (accepted paraphrases).
*   **`scripts/generator/generator_train/data/input/paraphrases.tsv`:** Input directory for generator training after a GAN loop iteration. Contains the original input data plus the accepted paraphrases.

### Model Input

*   **Generator Model:** The trained generator model is expected to be located in the `../generator/` directory. The exact location is determined by the `generator_generate.sh` and `generator_train.sh` scripts.
*   **Discriminator Model:** The trained discriminator model is expected to be located in the `../discriminator/` directory. The exact location is determined by the `discriminator_discriminate.sh` script.

## Scripts

*   **`1_gan_loop.sh`:** The main shell script that executes a single iteration of the GAN loop.
*   **`prepare_data()`:** A function within `1_gan_loop.sh` that handles data preparation for the generator and discriminator.

## Dependencies

*   **Generator Scripts:**
    *   `../generator/generator_generate/generator_generate.sh`: Script for generating paraphrases using the generator model.
    *   `../generator/generator_train/generator_train.sh`: Script for training the generator model.
*   **Discriminator Script:**
    *   `../discriminator/discriminator_discriminate/discriminator_discriminate.sh`: Script for discriminating phrases using the discriminator model.

## ASCII Art

The `gan_loop.ascii` file contains the ASCII art displayed at the beginning of the `1_gan_loop.sh` script.

## Note

This script runs a *single* iteration of the GAN loop. To run the loop continuously, you would typically use an external script or process to repeatedly call `1_gan_loop.sh`.

# 1_gan_loop

This program forever increases a dataset of human-like paraphrases from
pretrained paraphrase generator and phrase discriminator models.

The loop consists of the following steps:

- generates a batch of phrases
- discriminates which phrases it thinks is human-generated
- trains the generator on all phrases the discriminator thought was human-generated
- trains the discriminator to classify all generated phrases as not human-generated

```
GAN LOOP

   (train)
     |-------------<--------------    paraphrases_selected.tsv
     |                                          /
     v  paraphrases_generated.tsv              /
 GENERATOR  ---------------->  DISCRIMINATOR ->
     ^                                         \
     |                                          \
phrases_input.txt                               (discarded)
  (predict)
```

## Directory structure

Input is the trained models of generator and discriminator

```
# Input for generator generate
./data/input
|-phrases_input.txt
model/
|-<generator model>

# Input of discriminator discriminate
model/
|-<discriminator model>
