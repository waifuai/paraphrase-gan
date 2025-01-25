# 1_gan_loop

This program forever increases a dataset of human-like paraphrases from
pretrained paraphrase generator and phrase discriminator models.

The loop consists of the following steps:

- generates a batch of phrases
- discriminates which phrases it thinks is human-generated
- trains the generator on all phrases the discriminator thought was human-generated
- trains the discriminator to classify all generated phrases as not human-generated

```
GAN LOOP

   (train)
     |-------------<--------------    paraphrases_selected.tsv
     |                                          /
     v  paraphrases_generated.tsv              /
 GENERATOR  ---------------->  DISCRIMINATOR ->
     ^                                         \
     |                                          \
phrases_input.txt                               (discarded)
  (predict)
```

## Directory structure

Input is the trained models of generator and discriminator

```
# Input for generator generate
./data/input
|-phrases_input.txt
model/
|-<generator model>

# Input of discriminator discriminate
model/
|-<discriminator model>
