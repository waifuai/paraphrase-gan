# discriminator

Handles training and performing the discrimination of whether a given phrase
is human-generated or not.

- `discriminator_train` trains the discriminator.
- `discriminator_discriminate` performs the discrimination.
- `phrase_discriminator` specifies the instructions for `tensor2tensor`
as a `PhraseDiscriminatorProblem`.

```
DISCRIMINATOR TRAINING

paraphrases_generated.tsv
          v
     DISCRIMINATOR
```

```
DISCRIMINATOR DISCRIMINATION

paraphrases_generated.tsv                        paraphrases_selected.tsv
           |                                           /
           v                                          /
phrases_generated.txt ->  DISCRIMINATOR -> phrases_discrimination_labels.txt
                                                      \
                                                       \
                                                      (discarded)
