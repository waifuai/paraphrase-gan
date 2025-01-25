"""
Generates example data for paraphrases.
"""

import os

N_REPEAT_LINES = 5000
DATA_DIR = os.path.join(os.path.dirname(__file__), "data/input") # Use os.path.join for platform compatibility

os.makedirs(DATA_DIR, exist_ok=True)  # Ensure directory exists

def write_paraphrases(filepath, phrase_type):
  """Writes paraphrases to a file."""
  with open(filepath, 'w') as f: # 'w' mode overwrites, preventing infinite growth
    for _ in range(N_REPEAT_LINES):
      f.write(f"a human phrase\ta {phrase_type} paraphrase\t{phrase_type} paraphrase\n")
      f.write(f"another human phrase\tanother {phrase_type} paraphrase\n")

write_paraphrases(os.path.join(DATA_DIR, "paraphrases_selected.tsv"), "human")
write_paraphrases(os.path.join(DATA_DIR, "paraphrases_generated.tsv"), "machine")