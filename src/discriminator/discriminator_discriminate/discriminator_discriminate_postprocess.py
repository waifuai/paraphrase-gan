"""
Does the postprocessing of discrimate
"""

import os

cwd = os.path.dirname(os.path.realpath(__file__))

with open(cwd + '/data/output/phrases_discrimination_labels.tsv', 'r') as f_in:
  with open(cwd + '/data/output/phrases_discrimination_labels.tsv', 'r') as f_match:
    with open(cwd + '/data/output/paraphrases_selected.tsv', 'a') as f_out:
      label = f_in.readline()
      match = f_match.readline()
      while label and match:
        if label == "human":
          f_out.write(match)
        label = f_in.readline()
        match = f_match.readline()
