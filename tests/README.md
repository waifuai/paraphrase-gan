# Test Suite for GAN-based Paraphrase Generation

This README outlines the structure and purpose of the test suite for a GAN-based paraphrase generation system. The tests are designed to ensure the correctness of data preparation and training setup for both the discriminator and generator components of the GAN.

## Test Environment Setup

The test suite utilizes `pytest` and leverages fixtures to manage the test environment. Key fixtures include:

-   **`test_data`**: Provides sample input texts and their corresponding expected outputs (paraphrases). This fixture is scoped to the module, meaning it's created once per module.
-   **`clean_test_env`**: Creates a temporary directory (`gan_test`) within the system's temporary directory. This directory serves as a clean environment for each test, preventing interference from previous test runs. Pytest automatically cleans up this directory after the tests are executed.
-   **`test_dir`**: A fixture providing the path to a temporary directory ("temp\_test\_dir") within the tests' directory, used for setting up and running tests in an isolated environment. This is to ensure that test operations do not interfere with the actual codebase or data.

## Test Cases

The test suite is structured to cover different stages of the GAN pipeline, focusing primarily on data preparation. Each test function follows a pattern:

1. **Setup**: Setting up the necessary environment, including creating directories and copying required files.
2. **Execution**: Running the component under test (e.g., `DataPreparer`, `DiscriminatorProblem`, `GeneratorProblem`).
3. **Assertion**: Verifying the expected outcomes, such as the existence of processed data files.
4. **Cleanup**: Removing the test environment to ensure a clean state for subsequent tests.

### `test_0_gan_prep(test_dir)`

This test case verifies the functionality of the `DataPreparer` class, which is responsible for preparing the raw data for the GAN.

-   **Setup**: Creates a directory structure mimicking the expected input and output directories for the `DataPreparer`. It copies the necessary data from the `scripts/0_gan_prep/data` directory to the test environment.
-   **Execution**: Initializes and runs the `DataPreparer` with the configured input and output directories.
-   **Assertion**: Checks if the output directory exists after running `DataPreparer.prepare()`, indicating successful data preparation.
-   **Cleanup**: Removes the entire test directory.

### `test_1_gan_loop_discriminator_train_prep(test_dir)`

This test case checks the preparation of training data for the discriminator.

-   **Setup**: Sets up the test environment and creates a dummy `paraphrases_prepared.txt` file, which is required as input for `DiscriminatorProblem`.
-   **Execution**: Initializes `DiscriminatorProblem` and calls its `prepare()` method to generate the discriminator training data.
-   **Assertion**: Asserts that the `discriminator_data_prepared.jsonl` file exists after the preparation, indicating successful setup for discriminator training.
-   **Cleanup**: Cleans up the test environment.

### `test_1_gan_loop_generator_train_prep(test_dir)`

Similar to the discriminator test, this case verifies the preparation of training data for the generator.

-   **Setup**: Sets up the test environment and creates a dummy `paraphrases_prepared.txt` file.
-   **Execution**: Initializes `GeneratorProblem` and calls its `prepare()` method to generate the generator training data.
-   **Assertion**: Asserts that the `generator_data_prepared.jsonl` file exists after the preparation, confirming successful setup for generator training.
-   **Cleanup**: Cleans up the test environment.

## Running the Tests

To execute the test suite, ensure you have `pytest` installed. Then, navigate to the directory containing the test files (e.g., `tests/`) and run:

```bash
pytest
```

This command will discover and run all test functions within the `test_gan.py` file.

## Dependencies

-   `pytest`: Testing framework.
-   `pathlib`: For handling file paths.
-   `shutil`: For file operations like copying and removing directories.
-   `os`: For interacting with the operating system, like creating directories.
-   `sys`: For adding the project's root directory to the Python path, allowing imports from the `src` directory.

The project's code modules being tested (e.g., `DataPreparer`, `DiscriminatorProblem`, `GeneratorProblem`) are imported from the `src` directory. Ensure that the project structure is correctly set up so that these modules can be imported.
