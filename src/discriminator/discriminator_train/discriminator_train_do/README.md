# discriminator train do

Trains a discriminator to discriminate whether phrases are human-generated or not.
Replaces the previous discriminator model with a newly trained discriminator model.

```
DISCRIMINATOR TRAINING

paraphrases_generated.tsv
          v
     DISCRIMINATOR
```

## Directory structure

```
# Input
data
|-input
  |-t2t_data discriminator

# Output
model/
|-<discriminator model>
