# src

This directory contains the main source code for the paraphrase generation and discrimination project. It is organized into several subdirectories, each responsible for a specific aspect of the system.

## Directory Structure

The `src` directory has the following structure:

-   **`utils/`**: Contains utility modules for data preparation and other common tasks.
-   **`generator/`**: Houses the code for the paraphrase generator, including training and generation logic.
-   **`discriminator/`**: Contains the code for the discriminator, which determines whether a phrase is human-generated or machine-generated.

## Components

### `utils`

The `utils` directory provides helper functions and classes that are used throughout the project.

#### `prepare_data.py`

This script is responsible for preparing the data for both the generator and discriminator. It reads `.tsv` files from an input directory, performs cleaning and normalization, and writes the processed data to an output directory.

**Key Features:**

-   **Input Validation:** Ensures the input directory exists.
-   **Output Cleaning:** Clears the output directory before processing.
-   **File Processing:** Iterates through `.tsv` files, cleans each line (removes extra whitespace, converts to lowercase), and writes the cleaned data to new `.tsv` files.
-   **Configurable:** Allows customization of input and output directories through a configuration dictionary.

For a detailed explanation, refer to `src/utils/README.md`.

### `generator`

The `generator` directory contains the modules responsible for training and utilizing a paraphrase generator.

#### Key Files:

-   **`generator_train.py`**: Trains the paraphrase generator model.
-   **`generator_generate.py`**: Uses the trained generator to produce paraphrases for input phrases.
-   **`phrase_generator.py`**: Defines the `PhraseGenerationProblem` for `trax`, specifying the training and generation process.

**Workflow:**

1. **Training:** `generator_train.py` uses the `paraphrases_selected.tsv` dataset to train a `trax` model based on the instructions in `phrase_generator.py`.
2. **Generation:** `generator_generate.py` takes a list of input phrases from `phrases_input.txt`, feeds them to the trained generator, and saves the generated paraphrases to `paraphrases_generated.tsv`.

For a detailed explanation, refer to `src/generator/README.md`.

### `discriminator`

The `discriminator` directory houses the code for training and using a discriminator that classifies phrases as either human-generated or machine-generated.

#### Key Files:

-   **`discriminator_train.py`**: Trains the discriminator model.
-   **`discriminator_discriminate.py`**: Uses the trained discriminator to classify phrases.
-   **`phrase_discriminator.py`**: Defines the `PhraseDiscriminatorProblem` for `trax`, specifying the training and discrimination process.

**Workflow:**

1. **Training:** `discriminator_train.py` trains a `trax` model using the `paraphrases_generated.tsv` and `paraphrases_selected.tsv` datasets based on the instructions in `phrase_discriminator.py`.
2. **Discrimination:** `discriminator_discriminate.py` takes a list of phrases from `phrases_generated.txt`, passes them through the discriminator, and saves the classification labels (human or machine) to `phrases_discrimination_labels.txt`.

For a detailed explanation, refer to `src/discriminator/README.md`.

## Overall Workflow

The components in the `src` directory work together to create a system that can generate and discriminate paraphrases:

1. **Data Preparation:** `src/utils/prepare_data.py` prepares the raw data for training the generator and discriminator.
2. **Generator Training:** `src/generator/generator_train.py` trains the paraphrase generator.
3. **Paraphrase Generation:** `src/generator/generator_generate.py` generates paraphrases using the trained model.
4. **Discriminator Training:** `src/discriminator/discriminator_train.py` trains the discriminator to distinguish between human and machine-generated paraphrases.
5. **Discrimination:** `src/discriminator/discriminator_discriminate.py` classifies generated paraphrases using the trained discriminator.

This structure allows for modular development and independent testing of the generator and discriminator components.