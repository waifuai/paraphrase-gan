# Phrase Generator Trainer

This module trains a model to generate paraphrases for a given phrase using Trax, a deep learning library from Google.

## Overview

The core of this module is the `PhraseGeneratorProblem` class, which defines the problem of generating paraphrases and handles data loading and preprocessing. It uses a Transformer model from Trax, configured with specific hyperparameters for this task.

## Files

-   **`__init__.py`**: Standard Python module initialization file, imports the `problem` module.
-   **`problem.py`**: Contains the main logic for the phrase generator trainer.

    -   `PhraseGeneratorProblem` class:
        -   `approx_vocab_size`: Specifies the approximate vocabulary size for the model.
        -   `dataset_streams`: Defines how to create data streams for training and evaluation. It reads from a tab-separated file (`paraphrases_selected.tsv`) and handles splitting and shuffling.
        -   `generate_samples`: Generates sample pairs of phrases from the input file. It creates permutations of phrases within each line of the file, yielding input-target pairs for training.
    -   `transformer_phrase_generator`: Function that returns a set of hyperparameters for the Transformer model, optimized for the phrase generation task. These include:
        -   `num_encoder_layers`: Number of encoder layers in the Transformer.
        -   `num_decoder_layers`: Number of decoder layers in the Transformer.
        -   `d_model`: Dimensionality of the model's internal representations.
        -   `d_ff`: Dimensionality of the feedforward network within the Transformer.
        -   `n_heads`: Number of attention heads in the multi-head attention mechanism.
        -   `attention_dropout`: Dropout rate in the attention layers.
        -   `dropout`: General dropout rate.
        -   `learning_rate`: Learning rate for the optimizer.
    -   `transformer_phrase_generator_range`: Function that defines a range of hyperparameters for hyperparameter tuning. It allows for exploring different values for learning rate, number of layers, model dimensions, attention dropout, number of heads, and feedforward network dimensions.

## Usage

### Prerequisites

-   Python 3.x
-   TensorFlow
-   Trax
-   A dataset of phrase pairs in a tab-separated file named `paraphrases_selected.tsv`. The file should be located at `src/generator_train/generator_train_prep/data/input/paraphrases_selected.tsv` relative to the `problem.py` file.

### Training

1. **Prepare the dataset**: Ensure that the `paraphrases_selected.tsv` file is correctly formatted and located in the specified directory.
2. **Run the training script** that utilizes `PhraseGeneratorProblem` and `transformer_phrase_generator`. This will involve:
    -   Creating an instance of `PhraseGeneratorProblem`.
    -   Setting up the Transformer model using the defined hyperparameters.
    -   Creating data pipelines using `dataset_streams`.
    -   Training the model using Trax's training loop.

### Hyperparameter Tuning

To perform hyperparameter tuning, use the `transformer_phrase_generator_range` function to define the search space and utilize Trax's hyperparameter tuning capabilities to find the best combination of hyperparameters.

### Example of data format

The `paraphrases_selected.tsv` should contain lines of text separated by tabs. Each line could include more than two text phrases.

Example:

```
phrase1    phrase2    phrase3
another phrase1    another phrase2
```

This would generate the following input-target pairs:

```
(phrase1, phrase2)
(phrase2, phrase1)
(phrase1, phrase3)
(phrase3, phrase1)
(phrase2, phrase3)
(phrase3, phrase2)
(another phrase1, another phrase2)
(another phrase2, another phrase1)
```

## Notes

-   The module clears the TensorFlow FileWriter cache to ensure that TensorBoard events are properly logged.
-   The model configuration is smaller and uses more regularization compared to a typical translation model.
-   The code assumes a specific file path for the dataset relative to the location of `problem.py`. Adjust the path in `generate_samples` if your dataset is located elsewhere.
