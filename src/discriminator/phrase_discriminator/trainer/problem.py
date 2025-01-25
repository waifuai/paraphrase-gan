"""
Definition of the `trax` problem for the task of
`Discriminate whether a phrase is human vs machine generated`.
"""

import os
from trax.data import data_streams as streams
from trax.data import text_encoder
from trax import layers as tl
from trax.supervised import training
import tensorflow as tf

EOS = text_encoder.EOS

class PhraseDiscriminatorProblem():
  """Discriminate whether a phrase is human vs machine generated."""

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": training.Split.TRAIN,
        "shards": 1,
    }, {
        "split": training.Split.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2**14

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    del data_dir
    return ["not_human_phrase", "human_phrase"]

  def example_generator(self, filename):
    for idx, line in enumerate(tf.io.gfile.GFile(filename, "rb")):
      line = text_encoder.to_unicode_utf8(line.strip())
      if len(line.split("\t")) > 1:
        phrases = line.split("\t")
      else:
        continue

      yield {
          "inputs": phrases[0],
          "label": 1
      }

      for n_phrase in range(1, len(phrases)):
        yield {
            "inputs": phrases[n_phrase],
            "label": 0
        }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    filename = os.path.join(
        os.path.dirname(__file__),
        "../../../discriminator_train/discriminator_train_prep/data/input/paraphrases_generated.tsv"
    )

    for example in self.example_generator(filename):
      yield example

class PhraseDiscriminatorProblemCharacters(PhraseDiscriminatorProblem):
  """Character level settings of phrase discrimination problem"""

  @property
  def vocab_type(self):
    return text_encoder.VocabType.CHARACTER
