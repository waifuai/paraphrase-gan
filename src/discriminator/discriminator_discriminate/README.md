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
