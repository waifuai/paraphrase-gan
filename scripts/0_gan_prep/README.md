# gan prep

Prepares the GAN by training a paraphrase generator and phrase discriminator model.

```
GAN PREP

paraphrases_initial.tsv    phrases_human.txt
          |                        |
          v                        v
      GENERATOR              DISCRIMINATOR
                                   ^
                                   |
                          phrases_not_human.txt
```

## Directory structure

```
# Input
./data/input
|-paraphrases_initial.tsv

# Output of generator train
model/

# Output of discriminator train
model/
