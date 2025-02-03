# generator

Handles training and performing the generator of paraphrases to a corresponding input phrase.

- `generator_train` trains the generator.
- `generator_generate` performs the generation.
- `phrase_generator` specifies the instructions for `trax` as a
`PhraseGenerationProblem`.

```
GENERATOR TRAINING

paraphrases_selected.tsv
           |
           v
       GENERATOR
```

```
GENERATOR GENERATING

   GENERATOR ---> paraphrases_generated.tsv
       ^
       |
phrases_input.txt
    (predict)
