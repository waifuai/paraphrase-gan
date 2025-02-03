# Data Preparation Script

This Python script prepares data by reading `.tsv` files from an input directory, cleaning and normalizing them, and writing the processed data to an output directory.

## Script Overview

The core of the script is the `DataPreparer` class, which handles the following:

1. **Initialization:**
    *   Takes a configuration dictionary as input, specifying the input and output directories.
    *   Initializes `input_dir` and `output_dir` as `pathlib.Path` objects.

2. **Preparation (`prepare` method):**
    *   Validates that the input directory exists.
    *   Cleans the output directory (removes it if it exists and recreates it).
    *   Processes each `.tsv` file in the input directory.

3. **Input Validation (`_validate_inputs` method):**
    *   Checks if the input directory exists.
    *   Raises a `FileNotFoundError` if the input directory is not found.

4. **Output Cleaning (`_clean_output` method):**
    *   If the output directory exists, it's recursively removed using `shutil.rmtree`.
    *   The output directory is then created (along with any necessary parent directories).

5. **File Processing (`_process_files` and `_process_file` methods):**
    *   `_process_files` iterates over all `.tsv` files in the input directory.
    *   `_process_file` handles the processing of a single file:
        *   Opens the input and output files in UTF-8 encoding.
        *   Reads each line from the input file.
        *   Performs basic cleaning:
            *   Removes leading/trailing whitespace using `strip()`.
            *   Converts the line to lowercase using `lower()`.
            *   Skips empty lines or lines with less than two tab separated values.
            *   (This section can be extended for more complex data cleaning/normalization).
        *   Writes the cleaned line to the output file.
        *   Logs the processing of each file.

## Usage

1. **Prerequisites:**
    *   Python 3.x
    *   Make sure the input directory (default: `data/raw`) contains the `.tsv` files you want to process.

2. **Configuration:**
    *   The script uses a configuration dictionary. You can modify the `input_dir` and `output_dir` keys to specify your desired directories.

    ```python
    config = {
        'input_dir': 'data/raw',  # Your input directory
        'output_dir': 'data/processed/default'  # Your output directory
    }
    ```

3. **Running the script:**

    ```bash
    python src/utils/prepare_data.py
    ```

    This will:

    *   Process all `.tsv` files in the `input_dir`.
    *   Create the `output_dir` if it doesn't exist.
    *   Write the cleaned and normalized data to `.tsv` files in the `output_dir`, mirroring the names of the input files.
    *   Print "Data preparation complete." to the console upon successful completion.

## Example

If you have a file named `data/raw/example.tsv` with the following content:

```
Column1\tColumn2\tColumn3
  Hello World\tDATA\t123
  Another  Line\tMore Data\t456
```

After running the script, a new file named `data/processed/default/example.tsv` will be created with the following content:

```
column1\tcolumn2\tcolumn3
hello world\tdata\t123
another  line\tmore data\t456
```

## Notes

*   The script assumes that the input files are tab-separated value (TSV) files.
*   The cleaning and normalization process is currently basic. You can add more sophisticated logic within the `_process_file` method to handle specific data cleaning needs, such as removing special characters, handling different data types, etc.
*   The logging is set up to print informational messages to the console. You can configure logging further to write to a file or adjust the logging level.
*   Error handling is basic; only a `FileNotFoundError` for a missing input directory is implemented. Consider adding more robust error handling for file I/O issues, invalid data formats, etc.

