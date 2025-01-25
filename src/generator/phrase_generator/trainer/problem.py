"""
Generate a paraphrase for a given phrase.
"""
import os
import itertools
import tensorflow as tf

import trax
from trax import layers as tl
from trax.data import text_encoder, data_streams as streams
from trax.supervised import training

EOS = text_encoder.EOS
tf.summary.FileWriterCache.clear()  # ensure filewriter cache is clear for TensorBoard events file

class PhraseGeneratorProblem():
  """Generate a paraphrase for a given phrase."""

  def __init__(self, *args, **kwargs):
    self.EOS = EOS
    super().__init__(*args, **kwargs)

  @property
  def approx_vocab_size(self):
    return 2**16  # ~64k

  def dataset_streams(self, data_dir, dataset_split):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    split = dataset_split.TRAIN if dataset_split == training.Split.TRAIN else dataset_split.EVAL
    return streams.TextLineStream(
        os.path.join(data_dir, 'paraphrases_selected.tsv'),
        split=split,
        shuffle=dataset_split == training.Split.TRAIN)

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    filename = os.path.join(
        os.path.dirname(__file__),
        "../../../generator_train/generator_train_prep/data/input/paraphrases_selected.tsv"
    )
    with open(filename, 'r') as rawfp:
      for curr_line in rawfp:
        curr_line = curr_line.split('\t')
        if len(curr_line) >= 2:
          list_of_tuple_permutations_of_line = list(itertools.permutations(curr_line, 2))
          for i in range(len(list_of_tuple_permutations_of_line)):
            permutation_pair = list_of_tuple_permutations_of_line[i]
            yield {
              "inputs": permutation_pair[0],
              "targets": permutation_pair[1]
            }

# Smaller than the typical translate model, and with more regularization
def transformer_phrase_generator():
  hparams = trax.models.transformer.Transformer.default_hparams()
  hparams.num_encoder_layers = 2
  hparams.num_decoder_layers = 2
  hparams.d_model = 128
  hparams.d_ff = 512
  hparams.n_heads = 4
  hparams.attention_dropout = 0.6
  hparams.dropout = 0.6
  hparams.learning_rate = 0.05
  return hparams

# hyperparameter tuning ranges
def transformer_phrase_generator_range(rhp):
  rhp.set_float("learning_rate", 0.05, 0.25, scale=rhp.LOG_SCALE)
  rhp.set_int("num_encoder_layers", 2, 8)
  rhp.set_int("num_decoder_layers", 2, 8)
  rhp.set_discrete("d_model", [128, 256, 512])
  rhp.set_float("attention_dropout", 0.4, 0.7)
  rhp.set_discrete("n_heads", [2, 4, 8, 16, 32, 64, 128])
  rhp.set_discrete("d_ff", [512, 1024])
