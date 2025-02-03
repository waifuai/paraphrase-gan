# `discriminator_train_do` - Discriminator Training Script

This directory contains the script and resources necessary to train a discriminator model that distinguishes between human-generated and machine-generated phrases.

## Overview

The `discriminator_train_do.sh` script trains a Transformer-based discriminator model using the Trax library. It leverages a dataset of phrases (presumably including both human and machine-generated examples) to learn the distinguishing characteristics of each. The trained model is then saved for later use in other parts of the system, likely for evaluating the quality of generated text.

**Training Process**

The training pipeline can be visualized as follows:

```
paraphrases_generated.tsv (and other data)
          v
     DISCRIMINATOR TRAINING
          v
     Trained Discriminator Model
```

The script trains a model to distinguish input phrases (from `paraphrases_generated.tsv` and potentially other sources) based on whether they are human or machine-generated.

## Script Breakdown (`discriminator_train_do.sh`)

1. **Header and ASCII Art:**
    *   The script starts with a shebang `#!/bin/sh`, indicating it's a shell script.
    *   It includes a descriptive comment about its purpose.
    *   Prints a separator line and ASCII art to visually demarcate the script's execution.

2. **Variable Definitions:**
    *   `MODEL_NAME`: Sets the name of the model to "transformer\_phrase\_discriminator".
    *   `PROBLEM_NAME`:  Sets the problem name to "phrase\_discriminator\_problem" (likely a custom Trax problem definition).
    *   `DATA_DIR`: Specifies the directory containing the training data: `data/input`.
    *   `OUTPUT_DIR`: Specifies the directory where training outputs (e.g., logs, checkpoints) will be stored: `data/output`.
    *   `MODEL_DIR`: Specifies the directory where the final trained model will be saved: `model`.

3. **Directory Creation:**
    *   `mkdir -p "$OUTPUT_DIR"`: Creates the `data/output` directory if it doesn't exist.
    *   `mkdir -p "$MODEL_DIR"`: Creates the `model` directory if it doesn't exist.

4. **Trax Training Command:**
    *   `python -m trax.supervised.trainer`: Executes the Trax supervised trainer.
    *   `--output_dir="$OUTPUT_DIR"`: Directs Trax to store outputs in the specified output directory.
    *   `--model="$MODEL_NAME"`: Uses the "transformer\_phrase\_discriminator" model architecture.
    *   `--problem="$PROBLEM_NAME"`: Uses the "phrase\_discriminator\_problem" problem definition, which likely defines data loading and preprocessing steps.
    *   `--data_dir="$DATA_DIR"`: Tells Trax where to find the training data.
    *   `--train_steps="1000"`: Trains the model for 1000 steps.
    *   `--eval_steps="100"`: Evaluates the model every 100 steps.
    *   `--eval_frequency="100"`: Specifies how often the evaluation should be performed (every 100 steps in this case).

5. **Completion Message:**
    *   Prints a message indicating that the training is complete and where the model is saved.

## Directory Structure

```
discriminator_train_do/
├── data/
│   └── input/               # Input data directory
│       └── t2t_data          # Directory containing training data for Trax
│           └── discriminator # Data for the discriminator (likely text files)
│               └── paraphrases_generated.tsv # tab seperated file with data
└── model/                  # Output model directory
    └── <discriminator model>/ # Trained discriminator model files (created after training)
├── discriminator_train_do.sh  # Main training script
└── README.md               # This README file
└── disc_train_do.ascii     # ASCII art displayed by the script
```

**Input:**

*   `data/input/t2t_data/discriminator`: This directory contains the training data used by Trax. The specific format of the data depends on the `phrase_discriminator_problem` definition. It likely includes labeled examples of human-generated and machine-generated phrases. The file `paraphrases_generated.tsv` is likely one of these data files.

**Output:**

*   `model/<discriminator model>`: This directory will contain the trained discriminator model files after the `discriminator_train_do.sh` script is executed. The exact files depend on the Trax model implementation, but typically include model parameters, configuration files, and potentially checkpoints.
*   `data/output`: This directory will contain the logs and checkpoints created during the training process.

## Usage

To train the discriminator model, simply run the `discriminator_train_do.sh` script from the command line:

```bash
./discriminator_train_do.sh
```

**Prerequisites:**

*   **Python:** You need to have Python installed on your system.
*   **Trax:** The Trax library must be installed. You can install it using pip: `pip install trax`
*   **Data:** The necessary training data must be present in the `data/input/t2t_data/discriminator` directory. This likely requires prior setup steps that create the appropriate files within this directory.

## Notes

*   The script assumes that the `phrase_discriminator_problem` is defined elsewhere in the project. This problem definition would handle tasks such as data loading, preprocessing, and potentially defining the model's loss function and evaluation metrics.
*   The training parameters (e.g., `train_steps`, `eval_steps`) can be adjusted in the script to modify the training process.
*   This README assumes that there are prior steps in the overall project that produce the `paraphrases_generated.tsv` and any other files in `data/input/t2t_data/discriminator` directory.


# discriminator train do

Trains a discriminator to discriminate whether phrases are human-generated or not.
Replaces the previous discriminator model with a newly trained discriminator model.

```
DISCRIMINATOR TRAINING

paraphrases_generated.tsv
          v
     DISCRIMINATOR
```

## Directory structure

```
# Input
data
|-input
  |-t2t_data discriminator

# Output
model/
|-<discriminator model>
