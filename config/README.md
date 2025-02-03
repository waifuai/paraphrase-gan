# Project Directory Structure and Path Configuration

This repository contains a well-defined directory structure for managing a machine learning project, specifically one that likely involves a Generative Adversarial Network (GAN) based on the directory names.

## `paths.sh` Script

The `paths.sh` script is a crucial part of the project setup. It defines environment variables that represent absolute paths to various directories within the project. This ensures consistency and portability across different environments.

### Usage

Before running any other scripts in the project, source the `paths.sh` script into your current shell environment:

```bash
source config/paths.sh
```

This will make the defined environment variables available to other scripts and programs.

### Directory Structure

The script defines the following directory structure:

```
<PROJECT_ROOT>
├── config/           <- Configuration files
├── scripts/          <- Shell scripts for various tasks
├── src/              <- Source code (e.g., Python modules)
├── data/             <- Data storage
│   ├── raw/          <- Raw, unprocessed data
│   ├── processed/    <- Processed data ready for training
├── models/           <- Trained model files
│   ├── generator/    <- Saved generator models (in a GAN context)
│   ├── discriminator/<- Saved discriminator models (in a GAN context)
└── logs/             <- Log files for monitoring and debugging
```

### Environment Variables

The `paths.sh` script defines and exports the following environment variables:

| Variable Name        | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `PROJECT_ROOT`       | The absolute path to the root directory of the project. |
| `CONFIG_DIR`         | Path to the configuration directory.                  |
| `SCRIPTS_DIR`        | Path to the directory containing shell scripts.       |
| `SRC_DIR`            | Path to the source code directory.                    |
| `DATA_ROOT`          | Path to the root data directory.                      |
| `MODELS_ROOT`        | Path to the root directory for trained models.         |
| `LOGS_DIR`           | Path to the directory for log files.                 |
| `RAW_DATA_DIR`       | Path to the directory for raw data.                   |
| `PROCESSED_DATA_DIR` | Path to the directory for processed data.             |
| `GENERATOR_MODELS`   | Path to the directory for generator models.           |
| `DISCRIMINATOR_MODELS`| Path to the directory for discriminator models.       |

### Directory Creation

The `paths.sh` script also ensures that the necessary directories exist by creating them using `mkdir -p` if they are missing. The directories created are:

-   `logs/`
-   `data/raw/`
-   `data/processed/`
-   `models/generator/`
-   `models/discriminator/`

### Benefits

-   **Organization:**  A clear and consistent project structure makes it easier to manage code, data, models, and logs.
-   **Portability:** Using absolute paths defined by environment variables means your scripts will work correctly even if you move the project directory to a different location or if different users work on the project.
-   **Maintainability:**  It's much easier to update paths in a single script (`paths.sh`) than scattered throughout multiple files.
-   **Reproducibility:** This structure helps ensure that your project is well-organized, making it easier to reproduce results and collaborate with others.

### Notes

-   This project structure and the `paths.sh` script are especially suited for projects involving GANs, as it anticipates directories for saving generator and discriminator models. However, the structure can easily be adapted for other types of machine learning or deep learning projects.
-   The naming conventions used are very descriptive, which is good practice.
-   The script uses `"${BASH_SOURCE[0]}"` to determine the directory of the script itself, making it robust even when called from different locations.
