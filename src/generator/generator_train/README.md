# Generator Train

This component handles the preparation of data and the training of the generator model.

## Scripts

-   `generator_train_prep.sh`: Prepares the training data.
-   `generator_train_do.sh`: Performs the training of the generator model.

## Training Process

The training process involves the following steps:

1. **Input:** `paraphrases_selected.tsv` containing a dataset of paraphrases.
2. **Data Preparation:** The `generator_train_prep.sh` script prepares the data for training.
3. **Model Training:** The `generator_train_do.sh` script trains the generator model using the prepared data.
4. **Output:** The trained generator model is saved in the `model` directory.

## Directory Structure

```
# Input
data
|-input
  |-paraphrases_selected.tsv

# Output
model
|-<trained generator model>
