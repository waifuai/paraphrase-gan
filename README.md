# GAN-Based Paraphrase Generation (Hugging Face Transformers)

This project implements a Generative Adversarial Network (GAN) for generating paraphrases of phrases. It leverages the Hugging Face `transformers`, `datasets`, and `PyTorch` libraries for building and training the models.

**Generator Model:** `t5-small`
**Discriminator Model:** `bert-base-uncased`

## Project Structure

The project is organized as follows:

-   `src/`: Contains the main source code.
    -   `main.py`: The main script for running the GAN workflow. Includes mock data generation, data loading/preprocessing, model definition, initial training (`gan_prep_hf`), and the GAN training loop (`gan_loop_iteration_hf`).
    -   `tests/`: Contains unit tests.
        -   `test_gan.py`: Unit tests for various components (mock data, data loading, preprocessing, model loading, helper functions).
        -   `conftest.py`: Defines pytest fixtures for shared test resources (e.g., temporary directories).
-   `plans/`: Contains markdown files outlining development plans.
-   `lessons/`: Contains markdown files documenting lessons learned.
-   `data/`: Contains the data used for training and evaluation.
    -   `raw/`: Raw input data (mock TSV data is generated here: `mock_generator_input.tsv`, `mock_discriminator_input.tsv`).
    -   `processed/`: Processed data generated during the GAN loop.
        -   `generator/`: Files related to the generator.
            -   `generated/`: Raw generated phrases (`generated_phrases.txt`) and formatted TSV for discriminator input (`discriminator_input_generated.tsv`).
            -   `generator_combined_training_data.tsv`: Combined data for generator retraining.
        -   `discriminator/`: Files related to the discriminator.
            -   `classified/`: Generated phrases selected by the discriminator (`discriminator_selected_generated.tsv`).
-   `models/`: Stores trained models (saved in Hugging Face format).
    -   `generator/`: Stores generator models.
        -   `initial/`: Initial trained generator (`t5-small` fine-tuned).
        -   `latest/`: The generator model from the latest GAN iteration.
    -   `discriminator/`: Stores discriminator models.
        -   `initial/`: Initial trained discriminator (`bert-base-uncased` fine-tuned).
-   `logs/`: Stores log files (`run.log`) and training logs from `transformers.Trainer`.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8 or higher
    *   `pip` and `venv` (recommended)
    *   Git

2.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create and activate a virtual environment (recommended):**

    ```bash
    # Using standard venv
    python -m venv .venv
    # Activate:
    # Linux/macOS: source .venv/bin/activate
    # Windows: .venv\Scripts\activate

    # OR using uv (if installed)
    python -m uv venv .venv
    # Activate:
    # Linux/macOS: source .venv/bin/activate
    # Windows: .venv\Scripts\activate
    ```

4.  **Install dependencies:**
    *   Ensure `uv` is installed if using it (`pip install uv` or `python -m pip install uv`).
    *   Install requirements using `uv`:
        ```bash
        # Ensure the venv is activated first if uv isn't global
        python -m uv pip install -r requirements.txt
        ```
    *   The `requirements.txt` file includes: `pytest`, `transformers`, `datasets`, `accelerate`, `evaluate`, `torch`, `sentencepiece`, `protobuf`, `numpy`.

## Usage

### Running the GAN

The main script (`src/main.py`) orchestrates the entire GAN workflow using Hugging Face components. It performs the following steps:

1.  **Initialization:**
    *   Creates necessary directories (data, models, logs).
    *   Generates mock data TSV files (`mock_generator_input.tsv`, `mock_discriminator_input.tsv`) in `data/raw/` if they don't exist.
    *   Initializes Hugging Face tokenizers (`AutoTokenizer`) for the generator (`t5-small`) and discriminator (`bert-base-uncased`).
    *   Loads the mock data using `datasets.load_dataset`.
    *   Preprocesses (tokenizes) the datasets using `.map()`.
    *   Loads the base pre-trained models (`AutoModelForSeq2SeqLM`, `AutoModelForSequenceClassification`).

2.  **Initial Model Training (`gan_prep_hf`):**
    *   Defines `TrainingArguments` for both models.
    *   Defines data collators (`DataCollatorForSeq2Seq`, `DataCollatorWithPadding`).
    *   Trains the initial generator model on the mock generator data using `transformers.Trainer`. Saves the trained model to `models/generator/initial/` and copies it to `models/generator/latest/`.
    *   Trains the initial discriminator model on the mock discriminator data using `transformers.Trainer`. Saves the trained model to `models/discriminator/initial/`.

3.  **GAN Training Loop (`gan_loop_iteration_hf`):**
    *   This loop runs continuously until manually stopped (Ctrl+C).
    *   **Load Models:** Loads the latest generator (`models/generator/latest/`) and the initial discriminator (`models/discriminator/initial/`).
    *   **Prepare Generation Input:** Loads unique input phrases from the mock data.
    *   **Generate Phrases:** Uses `generator.generate()` to create paraphrases for the unique inputs. Saves raw generated text.
    *   **Prepare Discriminator Input:** Formats the generated phrases into a TSV file and loads it as a `Dataset`. Tokenizes the dataset.
    *   **Classify Phrases:** Uses `discriminator` model (via `Trainer.predict()`) to classify generated phrases (predicts 0 for machine-like, 1 for human-like).
    *   **Post-process:** Filters the generated phrases based on discriminator predictions (keeping those predicted as '1'). Pairs selected generated phrases with their original inputs and saves them to a TSV file (`discriminator_selected_generated.tsv`).
    *   **Combine Data:** Merges the original generator training data with the newly selected generated pairs, removing duplicates. Saves the combined data (`generator_combined_training_data.tsv`).
    *   **Retrain Generator:** Loads the combined dataset, tokenizes it, and retrains the generator model using `transformers.Trainer`. Saves the updated model back to `models/generator/latest/`.
    *   **(Future Work):** Optionally retrain the discriminator periodically.

To run the GAN, execute the following command from the project root directory:

```bash
# Ensure your virtual environment is activated
python src/main.py
```

The script will run indefinitely (or until an error occurs or it's stopped), performing the initial training and then iterating through the GAN loop. Stop with Ctrl+C.

### Testing

The `src/tests/` directory contains unit tests. To run the tests, use `pytest`:

```bash
# Ensure your virtual environment is activated
python -m pytest src/tests/
```

The tests cover:
- Mock data generation format.
- Data loading and preprocessing with `datasets`.
- Hugging Face model loading.
- Helper functions for post-processing and data combination (`postprocess_discriminator_output_hf`, `combine_data_hf`).
- Utility functions (`ensure_directory`).

## Core Components (Hugging Face Version)

-   **`main.py`:** Main script containing configuration, helper functions, and workflow orchestration.
-   **`CONFIG`:** Dictionary holding paths, filenames, model identifiers, and training parameters.
-   **`AutoTokenizer`:** Used for loading tokenizers for T5 and BERT.
-   **`load_dataset`:** Used for loading data from generated TSV files.
-   **`.map()`:** Used with custom preprocessing functions (`preprocess_generator`, `preprocess_discriminator`) to tokenize data.
-   **`AutoModelForSeq2SeqLM`:** Used for the T5 generator model.
-   **`AutoModelForSequenceClassification`:** Used for the BERT discriminator model.
-   **`TrainingArguments`:** Configures training parameters for the `Trainer`.
-   **`DataCollatorForSeq2Seq` / `DataCollatorWithPadding`:** Prepare batches for training.
-   **`Trainer`:** High-level API used for training and prediction.
-   **`train_hf_model`:** Helper function encapsulating the `Trainer` training process.
-   **`gan_prep_hf`:** Function to perform initial training of both models.
-   **`gan_loop_iteration_hf`:** Function executing one iteration of the GAN loop (generation, classification, data update, retraining).
-   **`generate_mock_paraphrases`:** Generates mock TSV data suitable for `load_dataset`.
-   **`postprocess_discriminator_output_hf`:** Filters generated phrases based on discriminator predictions.
-   **`combine_data_hf`:** Combines datasets for generator retraining.

## Key Changes from Trax Version

-   Replaced Trax models, problems, training loops, and decoding with Hugging Face equivalents (`transformers`, `datasets`, `Trainer`).
-   Uses PyTorch backend instead of TensorFlow/JAX (as configured).
-   Updated `requirements.txt`.
-   Adapted mock data generation and loading for `datasets`.
-   Rewrote preprocessing, training, generation, and classification steps using HF APIs.
-   Rewrote helper functions for post-processing and data combination.
-   Updated tests to match the new implementation.

## Further Improvements

-   **Implement Discriminator Training:** Train the discriminator within the GAN loop, potentially on a mix of real and generated data.
-   **Refine GAN Loop Logic:** Improve the interaction between generator and discriminator training (e.g., balance updates, use different loss functions).
-   **Metrics and Evaluation:** Add proper evaluation using metrics like BLEU, ROUGE for generation and Accuracy/F1 for classification during training and potentially within the loop. Use the `evaluate` library.
-   **Real Data:** Replace mock data generation with loading a standard paraphrase dataset (e.g., MRPC, PAWS) using `load_dataset`.
-   **Hyperparameter Tuning:** Optimize learning rates, batch sizes, epochs, etc.
-   **Configuration:** Move configuration from the Python dict to a separate file (e.g., YAML or JSON).
-   **Error Handling:** Add more specific error handling within the GAN loop.
-   **Command-Line Arguments:** Use `argparse` to allow configuration overrides via the command line.
-   **Dockerization:** Containerize the application.
