# GAN Preparation (0_gan_prep)

This directory contains scripts and resources for preparing the Generative Adversarial Network (GAN) used in this project. The preparation involves training two key components:

1. **Paraphrase Generator:** A model that learns to generate paraphrases of given phrases.
2. **Phrase Discriminator:** A model that learns to distinguish between human-written phrases and machine-generated (non-human) phrases.

## Process Overview

The `0_gan_prep.sh` script orchestrates the training process. It utilizes helper scripts and manages dependencies to ensure proper execution. Here's a breakdown of the steps:

1. **Initialization:**
    *   Sets up the necessary environment (paths, logging).
    *   Displays a cool ASCII art banner.

2. **Training the Generator:**
    *   Calls the `train_model` function to train the paraphrase generator.
    *   The `train_model` function handles potential training failures by implementing a retry mechanism (up to 3 retries with a 5-second delay).
    *   The generator training script is expected to be located at `../generator/generator_train/generator_train.sh`.
    *   Training data and model output will be managed within the `../generator/` directory.

3. **Training the Discriminator:**
    *   Similarly, calls `train_model` to train the phrase discriminator.
    *   The discriminator training script is expected to be located at `../discriminator/discriminator_train/discriminator_train.sh`.
    *   Training data and model output will be managed within the `../discriminator/` directory.

4. **Completion:**
    *   Prints a success message indicating the completion of GAN preparation.

## Data

### Input

The initial input data for training is located in `scripts/0_gan_prep/data/input/`. The `gen_mock_paraphrases.py` script is provided to generate mock data for testing purposes and creates the following files if run:

*   **`paraphrases_selected.tsv`:**  Contains tab-separated values representing human-generated paraphrases. Each line has the format: `human phrase \t human paraphrase \t human paraphrase` where the last two columns may contain the same paraphrase. It will have 10000 lines.
*   **`paraphrases_generated.tsv`:** Contains tab-separated values representing machine-generated paraphrases. Each line has the format: `human phrase \t machine paraphrase` or `human phrase \t machine paraphrase \t machine paraphrase` where the last two columns may contain the same paraphrase. It will have 10000 lines.

### Output

*   **Generator Model:** The trained generator model will be saved in the `../generator/` directory by the respective training script.
*   **Discriminator Model:** The trained discriminator model will be saved in the `../discriminator/` directory by the respective training script.

## Scripts

*   **`0_gan_prep.sh`:** The main shell script that orchestrates the GAN preparation process.
*   **`gen_mock_paraphrases.py`:** Python script to generate mock paraphrase data for testing.
*   **`test_0_gan_prep.sh`:** A shell script to test the `0_gan_prep.sh` script. It generates mock input data and then runs `0_gan_prep.sh`.

## Error Handling

*   **Robust Training:** The `train_model` function in `0_gan_prep.sh` implements error handling with retries to make the training process more robust.
*   **Logging:** The `log` function (sourced from `common.sh`) is used to provide informative messages during execution, including INFO, WARNING, and ERROR levels.
*   **Immediate Exit:** The `set -e` command ensures that the script exits immediately if any command fails.

## Dependencies

*   `scripts/common.sh`: A script containing common functions and variables, particularly the `log` function used for logging.
*   `scripts/paths.sh`: A script that defines path variables for the project.
*   Generator training script: `../generator/generator_train/generator_train.sh`
*   Discriminator training script: `../discriminator/discriminator_train/discriminator_train.sh`

## Testing

The `test_0_gan_prep.sh` script provides a basic test scenario. It generates mock paraphrase data using `gen_mock_paraphrases.py` and then executes `0_gan_prep.sh`.

## ASCII Art

The `gan_prep.ascii` file contains the ASCII art displayed at the beginning of the `0_gan_prep.sh` script. It doesn't serve any functional purpose other than being aesthetically pleasing.

# gan prep

Prepares the GAN by training a paraphrase generator and phrase discriminator model.

```
GAN PREP

paraphrases_initial.tsv    phrases_human.txt
          |                        |
          v                        v
      GENERATOR              DISCRIMINATOR
                                   ^
                                   |
                          phrases_not_human.txt
```

## Directory structure

```
# Input
./data/input
|-paraphrases_initial.tsv

# Output of generator train
model/

# Output of discriminator train
model/
