import pytest
import shutil
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.prepare_data import DataPreparer
from src.discriminator.phrase_discriminator.trainer.problem import (
    DiscriminatorProblem,
)
from src.generator.phrase_generator.trainer.problem import GeneratorProblem

def setup_test_environment(test_dir):
    """Setup a clean test environment."""
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    # Copy necessary data for testing
    shutil.copytree("./scripts/0_gan_prep/data", os.path.join(test_dir, "0_gan_prep", "data"))
    shutil.copytree(
        "./src/discriminator/phrase_discriminator/trainer",
        os.path.join(test_dir, "discriminator_trainer"),
    )
    shutil.copytree(
        "./src/generator/phrase_generator/trainer",
        os.path.join(test_dir, "generator_trainer"),
    )

def teardown_test_environment(test_dir):
    """Clean up the test environment."""
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_0_gan_prep(test_dir):
    setup_test_environment(test_dir)  # Setup before test

    # Prepare data for testing using DataPreparer
    config = {
        'input_dir': os.path.join(test_dir, "0_gan_prep", "data"),
        'output_dir': os.path.join(test_dir, "0_gan_prep", "data", "processed")
    }
    preparer = DataPreparer(config)
    preparer.prepare()

    # Assertions to check if data preparation was successful
    assert os.path.exists(
        os.path.join(test_dir, "0_gan_prep", "data", "processed")
    )

    teardown_test_environment(test_dir)  # Cleanup after test

def test_1_gan_loop_discriminator_train_prep(test_dir):
    setup_test_environment(test_dir)  # Setup before test

    # Create a dummy file for testing
    with open(os.path.join(test_dir, "0_gan_prep", "data", "paraphrases_prepared.txt"), "w") as f:
        f.write("This is a test.\n")

    # Prepare discriminator training data
    disc_problem = DiscriminatorProblem(
        os.path.join(test_dir, "discriminator_trainer"),
        os.path.join(test_dir, "0_gan_prep", "data", "paraphrases_prepared.txt"),
    )
    disc_problem.prepare()

    # Assertions to check if discriminator training data preparation was successful
    assert os.path.exists(
        os.path.join(
            test_dir,
            "discriminator_trainer",
            "data",
            "discriminator_data_prepared.jsonl",
        )
    )

    teardown_test_environment(test_dir)  # Cleanup after test

def test_1_gan_loop_generator_train_prep(test_dir):
    setup_test_environment(test_dir)  # Setup before test

    # Create a dummy file for testing
    with open(os.path.join(test_dir, "0_gan_prep", "data", "paraphrases_prepared.txt"), "w") as f:
        f.write("This is a test.\n")

    # Prepare generator training data
    gen_problem = GeneratorProblem(
        os.path.join(test_dir, "generator_trainer"),
        os.path.join(test_dir, "0_gan_prep", "data", "paraphrases_prepared.txt"),
    )
    gen_problem.prepare()

    # Assertions to check if generator training data preparation was successful
    assert os.path.exists(
        os.path.join(
            test_dir,
            "generator_trainer",
            "data",
            "generator_data_prepared.jsonl",
        )
    )

    teardown_test_environment(test_dir)  # Cleanup after test

@pytest.fixture(scope="module")
def test_dir():
    return os.path.join(os.path.dirname(__file__), "temp_test_dir")
