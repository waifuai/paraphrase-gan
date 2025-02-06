# GAN-Based Paraphrase Generation

This project implements a Generative Adversarial Network (GAN) for generating paraphrases of phrases. It leverages the Trax library for building and training the models.

## Project Structure

The project is organized as follows:

-   `src/`: Contains the main source code.
    -   `main.py`: The main script for running the GAN workflow.  Includes data preparation, model definition, training, and decoding logic.
    -   `tests/`: Contains unit tests.
        -   `test_gan.py`: Unit tests for various components of the GAN.
        -   `conftest.py`:  Defines pytest fixtures for shared test resources.
-   `config/`:  (Placeholder) Intended for configuration files (currently not used).
-   `scripts/`: (Placeholder) Intended for utility scripts (currently not used).
-   `data/`: Contains the data used for training and evaluation.
    -   `raw/`:  Raw input data (mock data is generated here).
    -   `processed/`:  Processed data, ready for Trax.
-   `models/`: Stores trained models.
    -   `generator/`: Stores trained generator models.
        - `initial/`: Initial trained generator.
        - `latest/`: The generator model from the latest GAN iteration.
    -   `discriminator/`: Stores trained discriminator models.
        - `initial/`: Initial trained discriminator.
-   `logs/`: Stores log files.

## Setup and Installation

1.  **Prerequisites:**
    -   Python 3.7 or higher
    -   pip
    -   virtualenv (recommended)

2.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create and activate a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    (Create `requirements.txt` with `pip freeze > requirements.txt` after installing trax, tensorflow, and pytest.)  Add the following lines to your `requirements.txt` file:
    ```
    trax
    tensorflow
    pytest
    ```

## Usage

### Running the GAN

The main script (`src/main.py`) orchestrates the entire GAN workflow.  It performs the following steps:

1.  **Initialization:**
    -   Creates necessary directories (data, models, logs).
    -   Generates mock data in `data/raw` if it doesn't already exist. This is crucial for initial setup and testing.  The mock data consists of simple, repeated phrases.
    -   Prepares the raw data into a format suitable for Trax (TSV files) in `data/processed`.

2.  **Initial Model Training (`gan_prep`):**
    -   Trains an initial generator model on the prepared data.
    -   Trains an initial discriminator model.
    -   These initial models are saved in `models/generator/initial` and `models/discriminator/initial`, respectively. This uses retries to handle potential training instability.

3.  **GAN Training Loop (`gan_loop_iteration`):**
    -   This loop runs continuously until manually stopped.
    -   **Generate Phrases:** Uses the latest generator (`models/generator/latest`) to generate paraphrases from the input data.  The output is written to `data/processed/generator/generated`.
    -   **Discriminate Phrases:**  The initial discriminator (`models/discriminator/initial`) classifies the generated phrases as "human" or "machine".  The output labels are saved to `data/processed/discriminator/classified`.
    -   **Post-process Discriminator Output:** Filters the generated paraphrases based on the discriminator's classifications.  Only phrases classified as "human" are kept and written to a new TSV file in `data/processed/discriminator/classified`.
    -   **Update Generator Training Data:** Combines the original generator training data with the newly accepted (filtered) paraphrases. This creates a new, augmented training dataset for the next generator iteration.
    -   **Train Generator:** Retrains the generator using the combined dataset. The updated generator model overwrites the previous `latest` model in `models/generator/`.

To run the GAN, execute the following command:

```bash
python src/main.py
```

The script will run indefinitely, continuously generating paraphrases, training the generator, and filtering outputs with the discriminator.  You can stop the script with Ctrl+C.

### Testing

The `src/tests/` directory contains unit tests for various components. To run the tests, use the following command:

```bash
pytest src/tests/
```

The tests use `pytest` and include fixtures (defined in `src/tests/conftest.py`) to set up a clean test environment in a temporary directory. The `clean_test_env` fixture creates a temporary directory and then ensures its cleanup after test execution, thanks to `pytest`'s built-in functionality.

Key tests include:

-   `test_data_preparer`: Verifies data cleaning and normalization.
-   `test_phrase_discriminator_problem`: Tests the Trax problem definition for the discriminator.
-   `test_phrase_generator_problem`: Tests the Trax problem definition for the generator.
-   `test_combine_data`: Tests the data combination function.
-   `test_postprocess_discriminator_output`: Tests the filtering of generated phrases.
-   `test_generate_mock_paraphrases`: Checks the mock data generation function.
-   `test_run_shell_command`: Tests the utility function for running shell commands.
-  `test_ensure_directory`: Tests the directory creation utility.

## Core Components

-   **DataPreparer:** Handles data loading, cleaning, and normalization.
-   **PhraseDiscriminatorProblem:**  Defines the Trax problem for the discriminator (classifying phrases as human or machine-generated).
-   **PhraseGeneratorProblem:** Defines the Trax problem for the generator (generating paraphrases).
-   **transformer_phrase_generator:** Defines the Transformer model architecture for the generator.
-   **transformer_phrase_discriminator:**  Defines the Transformer model architecture for the discriminator.
-   **train_model:** Trains a Trax model.
-   **decode_model:** Uses a trained Trax model for inference (generating or classifying phrases).
-   **gan_prep:** Trains the initial generator and discriminator models.
-   **gan_loop_iteration:** Executes a single iteration of the GAN training loop.
-   **postprocess_discriminator_output:** Filters generated paraphrases based on discriminator labels.
-   **combine_data:** Combines data from two TSV files, removing duplicates.
-   **generate_mock_paraphrases:** Generates mock paraphrase data for initial training and testing.

## Key Improvements and Explanations

-   **Clear Directory Structure:**  The directory structure is well-defined, separating data, models, logs, source code, and tests.  Constants define these paths for easy modification.
-   **Error Handling:**  The `run_shell_command` function provides robust error handling for shell commands, capturing output and raising exceptions on failure.
-   **Data Preparation:** The `DataPreparer` class encapsulates data preparation logic, making it reusable and testable.  It handles input validation, output cleaning, and file processing.
-   **Trax Problem Definitions:** The `PhraseDiscriminatorProblem` and `PhraseGeneratorProblem` classes define the Trax problems, encapsulating data loading and generation logic.
-   **Model Definitions:**  Separate functions define the Transformer model architectures for the generator and discriminator.
-   **Training and Decoding:**  The `train_model` and `decode_model` functions provide wrappers around Trax's training and decoding APIs.
-   **GAN Workflow:**  The `gan_prep` and `gan_loop_iteration` functions implement the core GAN workflow, including training the initial models and running the iterative training loop.
-   **Data Combination:** The `combine_data` function correctly combines data from multiple files, handling duplicates and missing files.
-   **Mock Data Generation:** The `generate_mock_paraphrases` function generates mock data, making it easy to start the GAN without requiring real data initially.
-   **Logging:**  A logger is set up to provide informative output and error messages.  Log levels (INFO for console, DEBUG for file) are used effectively.
-   **Testing:**  Comprehensive unit tests cover the key components of the project. The use of fixtures and a temporary directory ensures clean and isolated testing.
-   **Retry Mechanism:** The `gan_prep` function includes a retry mechanism for training, which is important for the potentially unstable training process of GANs.
-   **Clear Separation of Concerns:**  Functions and classes have well-defined responsibilities, making the code more modular and maintainable.
-   **Type Hinting:** Type hints are used throughout the code, improving readability and helping to catch errors early.
- **Continuous GAN Loop:** The `main` function sets up a `while True` loop, allowing continuous generation and refinement of the generator's output.
- **Idempotent Directory Creation**: The `ensure_directory` function is idempotent, meaning it doesn't cause errors if called multiple times on the same path.

## Further Improvements

-   **Configuration:**  Implement a configuration system (e.g., using YAML files) to manage hyperparameters and other settings.
-   **Metrics and Evaluation:**  Add more comprehensive metrics and evaluation methods, beyond simple accuracy.  Consider metrics like BLEU score for evaluating the quality of generated paraphrases.
-   **Discriminator Training:** Currently, only the initial discriminator is used in the GAN loop.  Implement training of the discriminator within the loop as well.
-   **Curriculum Learning:** Explore curriculum learning strategies to gradually increase the difficulty of the training data.
-   **Hyperparameter Tuning:**  Use Trax's hyperparameter tuning capabilities to find optimal hyperparameters for the models.
-   **Real Data:** Integrate with real paraphrase datasets.
-   **Command-Line Arguments:** Add command-line arguments to control various aspects of the script (e.g., training steps, data directories, model names).
-   **Dockerization:** Containerize the application using Docker for easier deployment and reproducibility.
