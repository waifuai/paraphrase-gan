# discriminator_discriminate

This component is part of a larger system (presumably related to paraphrase generation and evaluation) and focuses on **discriminating between human-generated and machine-generated phrases**. It uses a trained transformer model to classify phrases and then filters a set of generated paraphrases based on these classifications.

```ascii
     _ _               _           _             _   _
  __| (_)___  ___ _ __(_)_ __ ___ (_)_ __   __ _| |_(_)_ __   __ _
 / _` | / __|/ __| '__| | '_ ` _ \| | '_ \ / _` | __| | '_ \ / _` |
| (_| | \__ \ (__| |  | | | | | | | | | | | (_| | |_| | | | | (_| |
 \__,_|_|___/\___|_|  |_|_| |_| |_|_|_| |_|\__,_|\__|_|_| |_|\__, |
                                                             |___/
```

## Overview

The `discriminator_discriminate` component performs the following actions:

1. **Prepares input data:** Extracts the first column (phrases) from `paraphrases_generated.tsv` and saves it as `phrases_generated.tsv`.
2. **Runs a trained Trax model:** Uses a pre-trained transformer model (presumably trained to distinguish between human and machine-generated text) to classify each phrase in `phrases_generated.tsv`.
3. **Postprocesses the results:** Filters the original `paraphrases_generated.tsv` based on the model's classifications. It selects only those paraphrases where the corresponding phrase was classified as "human."
4. **Outputs filtered paraphrases:**  Saves the selected paraphrases to `paraphrases_selected.tsv`.

## Data Flow and Processing

```
DISCRIMINATOR DISCRIMINATION

paraphrases_generated.tsv                        paraphrases_selected.tsv
           |                                           /
           v                                          /
phrases_generated.txt ->  DISCRIMINATOR -> phrases_discrimination_labels.txt
                                                      \
                                                       \
                                                      (discarded)
```

### Input

The primary input is `data/input/paraphrases_generated.tsv`, a tab-separated file with the following format:

```
a phrase<tab>another paraphrase
a phrase<tab>another paraphrase of same paraphrase
again a phrase<tab>again another paraphrase
```

### Intermediate Files

1. **`data/input/phrases_generated.tsv`**: A single-column file containing only the first phrase from each line of `paraphrases_generated.tsv`.

    ```
    a phrase
    a phrase
    again a phrase
    ```

2. **`data/output/phrases_discrimination_labels.tsv`**: Contains the model's classification for each phrase in `phrases_generated.tsv`.

    ```
    not_human
    human
    human
    ```

### Output

The final output is `data/output/paraphrases_selected.tsv`, containing only the rows from `paraphrases_generated.tsv` where the corresponding phrase was classified as "human" by the discriminator.

```
a phrase<tab>another paraphrase of same phrase
again a phrase<tab>again another paraphrase
```

## Directory Structure

```
data/
├── input/
│   └── paraphrases_generated.tsv
└── output/
    └── paraphrases_selected.tsv
```

## Scripts

### `discriminator_discriminate.sh`

This script orchestrates the entire discrimination process.

1. **Setup:**
    *   Creates necessary directories (`data/input`, `data/output`).
    *   Defines paths to the model, problem definition, input data, and output directories.

2. **Data Preprocessing:**
    *   Extracts the first column (phrases) from `paraphrases_generated.tsv` to create `phrases_generated.tsv`.

3. **Model Decoding (Discrimination):**
    *   Runs the `trax.supervised.decode` command with the following key parameters:
        *   `--model`: Specifies the name of the transformer model (`transformer_phrase_discriminator`).
        *   `--problem`: Specifies the name of the problem definition (`phrase_discriminator_problem`).
        *   `--data_dir`: Points to the directory containing the input data (`data/input`).
        *   `--output_dir`: Specifies the directory for output files (`data/output`).
        *   `--decode_steps="100"`: (Likely) sets a limit on the decoding steps, though for classification, this might not be relevant.
        *   `--decode_in_memory=True`: Loads the entire dataset into memory for faster processing (suitable if the dataset is not too large).

4. **Output Renaming:**
    *   Renames the Trax output file from `decode_out.txt` to `phrases_discrimination_labels.tsv`.

5. **Postprocessing:**
    *   Executes the `discriminator_discriminate_postprocess.py` script to filter the paraphrases.

### `discriminator_discriminate_postprocess.py`

This Python script performs the postprocessing of the discrimination results.

1. **Reads Files:** Opens three files:
    *   `phrases_discrimination_labels.tsv` (input, model's classifications).
    *   `paraphrases_generated.tsv` (input, original paraphrases, opened as 'f\_match').
    *   `paraphrases_selected.tsv` (output, filtered paraphrases, in append mode).

2. **Filtering Logic:** Iterates through the classification labels and the original paraphrases simultaneously.
    *   If a label is "human," the corresponding line from `paraphrases_generated.tsv` is written to `paraphrases_selected.tsv`.
    *   This effectively filters out paraphrases associated with phrases classified as "not\_human."

## Usage

To run the discrimination process, execute the `discriminator_discriminate.sh` script:

```bash
./discriminator_discriminate.sh
```

**Prerequisites:**

*   A trained Trax model named `transformer_phrase_discriminator` must exist in the `model/` directory.
*   A problem definition named `phrase_discriminator_problem` must be defined (likely in a separate Python file that Trax can access).
*   The input file `paraphrases_generated.tsv` must be present in the `data/input/` directory.
*   The `trax` library must be installed and accessible.

## Notes

*   The specific details of the Trax model training and problem definition are not included in this component, but they are essential for its correct functioning.
*   The `--decode_steps` parameter in the Trax command might need adjustment or clarification depending on the specific configuration of the discriminator model.
*   The script assumes that the number of lines in `phrases_discrimination_labels.tsv` and `paraphrases_generated.tsv` are the same and that the lines correspond to each other.

This README provides a detailed explanation of the `discriminator_discriminate` component. It clarifies the purpose, data flow, processing steps, and usage instructions. It also highlights the dependencies and assumptions made by the scripts. Remember to adapt this README if you make any changes to the code or the workflow.



# discriminator discriminate

Handles preparation and performing the discrimination of whether a give
phrase is human-generated or not.

- `discriminator_discriminate` processes the data and performs the discrimination.

```
DISCRIMINATOR DISCRIMINATION

paraphrases_generated.tsv                        paraphrases_selected.tsv
           |                                           /
           v                                          /
phrases_generated.txt ->  DISCRIMINATOR -> phrases_discrimination_labels.txt
                                                      \
                                                       \
                                                      (discarded)
```

### Directory structure

```
data
|-input
  |-paraphrases_generated.tsv
|-output
  |-paraphrases_selected.tsv
```

### File processing

#### Preprocessing

The program initially receives the file `paraphrases_generated.tsv` which is in the format:

```
a phrase<tab>another paraphrase
a phrase<tab>another paraphrase of same paraphrase
again a phrase<tab>again another paraphrase
```

It then processes it into the `phrases_generated.txt` in the format:

```
a phrase
a phrase
again another phrase
```

#### Postprocessing

After the discrimination of phrases we have the file `phrases_discrimination_labels.tsv` in the format:

```
not_human
human
human
```

We then process this file into `paraphrases_selected.tsv` by selecting the rows of `paraphrases_generated` which have corresponding `human` rows in `phrases_discrimination_labels.tsv` in the format:

```
a phrase<tab>another paraphrase of same phrase
again a phrase<tab>again another paraphrase
