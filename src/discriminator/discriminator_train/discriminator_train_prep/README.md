# Discriminator Train Prep

This component prepares a dataset for training a discriminator model. The goal of this discriminator is to determine whether a given phrase is human-generated or machine-generated.

## Overview

The process begins with a TSV file (`paraphrases_generated.tsv`) containing pairs of phrases, likely a phrase and its paraphrase. This data is then processed by the `discriminator_train_prep.sh` script to generate a dataset suitable for training with the Trax library.

The diagram below illustrates the flow:

```
DISCRIMINATOR TRAINING

paraphrases_generated.tsv
            v
      DISCRIMINATOR
```

## Script: `discriminator_train_prep.sh`

This shell script orchestrates the data preparation process. Here's a breakdown of its functionality:

1. **Prints a header and a visual ASCII art from `disc_train_prep.ascii`.**
2. **Clears and creates the output directory:** Ensures a clean output directory at `data/output`.
3. **Executes Trax data generation:** Utilizes the `trax.data.tf_inputs` module to generate the training dataset.
    *   `--output_dir`: Specifies the output directory as `data/output`.
    *   `--params`: Sets the problem name to `phrase_discriminator_problem`. This likely refers to a predefined configuration within Trax.
    *   `--n_shards`: Sets the number of output shards to 1.
    *   `--input_dir`: Specifies the input directory containing `paraphrases_generated.tsv` as `data/input`.

## Data

### Input: `data/input/paraphrases_generated.tsv`

This tab-separated values file contains pairs of phrases. The structure is simple:

```tsv
phrase1\tparaphrase1
phrase2\tparaphrase2
phrase3\tparaphrase3
```

*   Each line represents a pair.
*   The first column (`phrase1`, `phrase2`, etc.) likely contains the original phrase.
*   The second column (`paraphrase1`, `paraphrase2`, etc.) likely contains a paraphrase of the corresponding phrase.

### Output: `data/output`

This directory will contain the generated dataset after running `discriminator_train_prep.sh`. The exact format of the output files is determined by the `trax.data.tf_inputs` module and the configuration specified by `phrase_discriminator_problem`.

## Directory Structure

```
src/discriminator/discriminator_train/discriminator_train_prep/
├── data
│   ├── input
│   │   └── paraphrases.tsv  # Input TSV file with phrase pairs
│   └── output              # Output directory for the generated dataset
├── discriminator_train_prep.sh # Main script for data preparation
├── disc_train_prep.ascii       # ASCII art displayed by the script
└── README.md                   # This README file
```

## Usage

To use this component, simply execute the `discriminator_train_prep.sh` script:

```bash
./discriminator_train_prep.sh
```

This will generate the training dataset in the `data/output` directory.


# discriminator train prep

Prepares dataset for training of a discriminator to discriminate whether
phrases are human-generated or not.

```
DISCRIMINATOR TRAINING

paraphrases_generated.tsv
            v
      DISCRIMINATOR
```

## Directory structure

```
data
|-input
  |-paraphrases_generated.tsv
|-output
  |-
