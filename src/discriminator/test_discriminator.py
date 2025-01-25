"""
Test file for disciminator
"""

import subprocess
import os


def sh(command):
  # run shell script to behave independently of which directory it is run from
  subprocess.call(
    command.split(" "),
    cwd=os.path.dirname(os.path.realpath(__file__))
  )


def sh_get(command):
  # run shell script to behave independently of which directory it is run from
  return subprocess.check_output(command.split(" "), cwd=os.path.dirname(os.path.realpath(__file__)))


def test_disciminator_train_do():
  # prepare the test files to be consumed
  sh("sh test_discriminator_train_prep.sh")
  cwd = os.path.dirname(os.path.realpath(__file__))

  print("Running from test_discriminator_train.py")
  assert sh_get("ls " + cwd + "/discriminator_train/discriminator_train_prep/data/output/") == b"t2t_data\n"

  sh("sh test_discriminator_train_do.sh")

  assert os.path.exists(f"{cwd}/discriminator_train/discriminator_train_do/model")


def test_disciminator_discriminate():
  sh("sh test_discriminator_discriminate.sh")
  # this works when running python3 script.py
  # but if you run directly it will say NameError: name '__file__' is not defined
  cwd = os.path.dirname(os.path.realpath(__file__))
  # eg cwd is /foo/bar
  print("cwd = ", cwd)

  with open(cwd + "/discriminator_discriminate/data/output/phrases_discrimination_labels.txt", 'r') as myfile:
    data = myfile.readline()
    assert data == "human_phrase\n" or data == "not_human_phrase\n"
    data = myfile.readline()
    assert data == "human_phrase\n" or data == "not_human_phrase\n"
    data = myfile.readline()
    assert data == "human_phrase\n" or data == "not_human_phrase\n"
