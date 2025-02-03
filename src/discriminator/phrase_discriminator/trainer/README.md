# Phrase Discriminator Trainer

This code defines a `trax` problem for training a model to discriminate between human-generated and machine-generated phrases. It provides the necessary components for data generation, encoding, and training within the `trax` framework.

## Overview

The core of this code is the `PhraseDiscriminatorProblem` class, which inherits from `trax`'s problem definition. It specifies how to generate training and evaluation data, handle vocabulary, and define class labels for the task of phrase discrimination.

### Key Features:

-   **Data Generation:**
    -   `example_generator`: Reads a tab-separated file (`paraphrases_generated.tsv`) where the first column is assumed to be a human-generated phrase and subsequent columns are machine-generated paraphrases.
    -   `generate_samples`: Yields examples with "inputs" (the phrase) and "label" (1 for human, 0 for machine).
-   **Vocabulary:**
    -   `approx_vocab_size`: Sets an approximate vocabulary size of 2^14.
    -   `PhraseDiscriminatorProblemCharacters`: A subclass that uses character-level encoding instead of subword encoding.
-   **Class Labels:**
    -   `num_classes`: Defines 2 classes for the task.
    -   `class_labels`: Provides human-readable labels: "not_human_phrase" and "human_phrase".
-   **Dataset Splits:**
    -   `dataset_splits`: Specifies how to split the data into training and evaluation sets.
    -   `is_generate_per_split`: Indicates that data generation is needed for each split.

### Input Data Format

The input data is expected to be in a tab-separated file named `paraphrases_generated.tsv`. Each line should follow this format:

```
<human_phrase>\t<machine_paraphrase_1>\t<machine_paraphrase_2>\t...
```

-   The first column (`<human_phrase>`) represents a human-generated phrase.
-   Subsequent columns (`<machine_paraphrase_1>`, `<machine_paraphrase_2>`, etc.) represent machine-generated paraphrases of the human phrase.

### Usage

This code is intended to be used within the `trax` framework for training a model. It defines the problem, and other `trax` components can be used to build the model, configure training, and evaluate performance.

For instance, you can use the `PhraseDiscriminatorProblem` or the `PhraseDiscriminatorProblemCharacters` class as the `problem` argument when creating a `trainer.Trainer` instance to train a model on this task.

**Note**: The exact location of `paraphrases_generated.tsv` is relative to the script. You may need to adjust it based on your project's directory structure.
