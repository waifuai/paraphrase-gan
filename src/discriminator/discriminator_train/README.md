## Discriminator Train

This component handles the preparation and training of a discriminator model. The discriminator's task is to determine whether a given phrase is human-generated or not.

## Scripts

-   `discriminator_train_prep.sh`: Prepares the training data.
-   `discriminator_train_do.sh`: Performs the training of the discriminator model.

## Training Process

The training process involves the following steps:

1. **Input:** `paraphrases_generated.tsv` containing a dataset of paraphrases.
2. **Data Preparation:** The `discriminator_train_prep.sh` script prepares the data for training.
3. **Model Training:** The `discriminator_train_do.sh` script trains the discriminator model using the prepared data.
4. **Output:** The trained discriminator model is saved in the `model` directory.

## Directory Structure

```
# Input
data
|-input
  |-paraphrases_generated.tsv

# Output
model
|-<trained discriminator model>
