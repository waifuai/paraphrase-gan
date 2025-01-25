"""
Generates an example of paraphrases_selected.tsv and paraphrases_generated.tsv
"""

import os

# The number of times to repeat the two lines in the generated files
N_REPEAT_LINES = 5000
cwd = os.path.dirname(os.path.realpath(__file__))

with open(cwd + '/data/input/paraphrases_selected.tsv', 'a') as myfile:
  for i in range(N_REPEAT_LINES):
    myfile.write("a human phrase\ta human paraphrase\thuman paraphrase\n")
    myfile.write("another human phrase\tanother human paraphrase\n")

with open(cwd + '/data/input/paraphrases_generated.tsv', 'a') as myfile:
  for i in range(N_REPEAT_LINES):
    myfile.write("a human phrase\ta machine phrase\tmachine phrase\n")
    myfile.write("another human phrase\tanother machine phrase\n")
