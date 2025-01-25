"""
Test file for the training of the generator, including the test for the associated data preparation step
"""

import subprocess
import os


def sh(command):
  # run shell script to behave independently of which directory it is run from
  subprocess.call(command.split(" "), cwd=os.path.dirname(os.path.realpath(__file__)))


def sh_get(command):
  # run shell script to behave independently of which directory it is run from
  return subprocess.check_output(command.split(" "), cwd=os.path.dirname(os.path.realpath(__file__)))


def test_generator_train_prep():
  sh("sh test_generator_train_prep.sh")
  # this works when running python3 script.py
  # but if you run directly it will say NameError: name '__file__' is not defined
  cwd = os.path.dirname(os.path.realpath(__file__))
  # eg cwd is /foo/bar
  print("cwd = ", cwd)

  assert sh("ls " + cwd + "/generator_train/data/output/") == "t2t_data"


def test_generator_train_do():
  # assumes the test_generator_train_prep.sh is run by test above
  # run training on the test files that were prepared
  # WARNING: the test will probably fail because the test data is too small
  # need at least about 1000+ test data, otherwise it would cause iteration error on tpu
  sh("sh test_generator_train.sh")
  # this works when running python3 script.py
  # but if you run directly it will say NameError: name '__file__' is not defined
  cwd = os.path.dirname(os.path.realpath(__file__))
  # eg cwd is /foo/bar
  print("cwd = ", cwd)

  assert os.path.exists(f"{cwd}/generator_train/generator_train_do/model")
