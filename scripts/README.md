# scripts

This directory contains a collection of shell scripts designed to manage and automate various aspects of the project, including data preparation, model training, GAN preparation, and the main GAN training loop. It's further organized into subdirectories, each with a specific purpose and its own `README.md` file for detailed documentation.

## Overview

The scripts are organized to provide a modular and reusable approach to common tasks. They leverage a shared configuration and logging mechanism for consistency and ease of maintenance. The primary focus is on setting up and training a Generative Adversarial Network (GAN) consisting of a paraphrase generator and a phrase discriminator.

## Key Scripts and Directories

### `common.sh`

This script acts as a central hub for shared configurations, functions, and error handling logic.

**Features:**

*   **Unified Error Handling:** Implements a `trap` to catch errors and uses the `handle_error` function to log error messages with line numbers and exit the script.
*   **Logging:** Provides a `log` function to standardize logging messages with timestamps and log levels (e.g., INFO, ERROR). Logs are appended to `run.log` in the specified `LOG_DIR`.
*   **Base Directory and Log Directory:** Defines `BASE_DIR` (project root) and `LOG_DIR` (`logs` subdirectory within the project root) for consistent path handling.
*   **`load_config` Function:** Loads configuration variables, specifically paths, from `config/paths.sh`. This function centralizes the management of paths used throughout the scripts.
*   **`prepare_model_data` Function:** A common function for preparing data for different model types and phases. It uses `prepare_data.py` utility script with parameters like model type, phase, input directory, and output directory.

**Dependencies:**

*   `config/paths.sh`: Configuration file defining project paths.
*   `src/utils/prepare_data.py`: Python script responsible for data preparation.

### `prepare_data.sh`

This script provides a simplified interface for preparing data for different models and phases.

**Features:**

*   **`prepare_data` Function:** Takes `model_type` (e.g., 'generator', 'discriminator') and `phase` (e.g., 'train', 'eval') as arguments. It loads the configuration and calls the `prepare_model_data` function from `common.sh` to perform the data preparation.

**Usage Example:**

```bash
./prepare_data.sh generator train
```

This command would prepare the training data for the 'generator' model.

### `train_model.sh`

This script handles the training process for different machine learning models using the Trax library.

**Features:**

*   **`train_model` Function:** Takes `model_type`, `config_file`, and other training-related parameters.
    *   Loads the project configuration using `load_config`.
    *   Logs the start of the training process using the `log` function.
    *   Executes the `trax.supervised.trainer` command with parameters like:
        *   `problem`: The name of the problem being addressed.
        *   `model`: The architecture of the model.
        *   `data_dir`: The directory containing processed data.
        *   `output_dir`: The directory to save the trained model.
        *   `config_file`: Path to a configuration file specifying model parameters.
        *   `train_steps`: Number of training steps.
        *   `eval_frequency`: Frequency of evaluation steps.
    *   Logs the successful completion of training.

**Usage Example:**

```bash
./train_model.sh generator config/generator_config.gin
```

This command would start the training process for the 'generator' model using the specified configuration file.

### `0_gan_prep/`

This directory contains scripts and resources for the initial preparation of the GAN. It focuses on training the two key components: the paraphrase generator and the phrase discriminator.

*   **`0_gan_prep.sh`:** The main script that orchestrates the preparation process.
*   **`gen_mock_paraphrases.py`:** A Python script to generate mock paraphrase data for testing.
*   **`test_0_gan_prep.sh`:**  A script to test `0_gan_prep.sh`.
*   **`data/`:** Contains input data and generated mock data.
*   **`README.md`:** Provides detailed documentation of the GAN preparation process, including the steps involved, data used, dependencies, and error handling.

**For a comprehensive understanding of the GAN preparation process, refer to `scripts/0_gan_prep/README.md`.**

### `1_gan_loop/`

This directory will contain the scripts and resources related to the main GAN training loop. (This is a placeholder, as the actual implementation is not provided in the given code. However, it's a crucial part of the described GAN setup.)

**Expected Contents (Based on the provided context):**

*   **`1_gan_loop.sh`:** The main script implementing the iterative GAN training loop.
*   **`README.md`:** Detailed documentation explaining the GAN training loop, its logic, dependencies, and expected behavior.

**Expected Behavior of `1_gan_loop.sh`:**

*   **Iterative Training:** The script would likely alternate between training the generator and discriminator for a certain number of steps or epochs.
*   **Adversarial Interaction:** The generator would be trained to produce paraphrases that can fool the discriminator, while the discriminator would be trained to distinguish between real (human-generated) and fake (generator-generated) paraphrases.
*   **Evaluation:** The script might include steps to evaluate the performance of the generator and discriminator periodically.
*   **Model Saving:** The script should save the models at various checkpoints or after a certain number of iterations.

## Error Handling

The scripts are designed with robustness in mind. They utilize:

*   **`set -Eeuo pipefail`:** Ensures that scripts exit immediately if any command fails or if any unset variable is referenced.
*   **`trap 'handle_error $LINENO' ERR`:** Catches errors and executes the `handle_error` function, which logs the error with the line number and exits.

## Dependencies

*   **Trax:** The scripts utilize the Trax library for model training. Ensure that Trax is installed and properly configured in your environment.
*   **Python 3:** The `prepare_model_data` function calls a Python script (`prepare_data.py`), so Python 3 should be available.
*   **`config/paths.sh`:** This configuration file, sourced by `common.sh`, defines essential paths used throughout the project. Make sure this file is correctly set up.

## Configuration

Before running the scripts, make sure to:

1. **Set up `config/paths.sh`:** This file should define necessary paths for your project, such as data directories, model output directories, etc.
2. **Install dependencies:** Install Trax and any other required Python packages.

## Logging

The `log` function in `common.sh` provides a consistent way to log messages to both the console and the `run.log` file. This helps in monitoring the execution of scripts and debugging any potential issues. The log file is located at `${LOG_DIR}/run.log`, where `LOG_DIR` is defined in `common.sh` (defaults to `logs` in the project root).