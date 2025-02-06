# test_gan.py (Test Cases)

def test_data_preparer(clean_test_env):
    """Tests the DataPreparer class."""
    input_dir = clean_test_env / "input"
    output_dir = clean_test_env / "output"
    ensure_directory(input_dir)

    # Create a sample input file
    with open(input_dir / "test.tsv", "w", encoding="utf-8") as f:
        f.write("Phrase 1\tParaphrase 1\n")
        f.write("  Phrase 2  \t  Paraphrase 2  \n")  # Extra whitespace
        f.write("Phrase 3\tParaphrase 3\tExtra Column\n")  # Extra column
        f.write("\n") #Empty Line
        f.write("Phrase4\t") #Missing paraphrase

    preparer = DataPreparer(input_dir, output_dir)
    preparer.prepare()

    assert (output_dir / "test.tsv").exists()
    with open(output_dir / "test.tsv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert lines == [
        "phrase 1\tparaphrase 1\n",
        "phrase 2\tparaphrase 2\n",
        "phrase 3\tparaphrase 3\textra column\n",
    ]

def test_phrase_discriminator_problem(clean_test_env):
    """Tests the PhraseDiscriminatorProblem class."""
    data_dir = clean_test_env / "data"
    ensure_directory(data_dir)

    # Create a dummy input file
    with open(data_dir / "paraphrases_generated.tsv", "w", encoding="utf-8") as f:
        f.write("human phrase 1\tmachine paraphrase 1\n")
        f.write("human phrase 2\tmachine paraphrase 2\n")

    problem = PhraseDiscriminatorProblem(data_dir=str(data_dir))
    examples = list(problem.generate_samples(str(data_dir), None, None))
    assert len(examples) == 4  # 2 phrases * 2 examples (1 human, 1 machine)
    assert examples[0] == {"inputs": "human phrase 1", "label": 1}
    assert examples[1] == {"inputs": "machine paraphrase 1", "label": 0}
    assert examples[2] == {"inputs": "human phrase 2", "label": 1}
    assert examples[3] == {"inputs": "machine paraphrase 2", "label": 0}


def test_phrase_generator_problem(clean_test_env):
    """Tests the PhraseGeneratorProblem class."""

    data_dir = clean_test_env / "data"
    ensure_directory(data_dir)

     # Create a dummy input file
    with open(data_dir / "paraphrases_selected.tsv", "w", encoding="utf-8") as f:
        f.write("phrase 1\tparaphrase 1\n")
        f.write("phrase 2\tparaphrase 2a\tparaphrase 2b\n")

    problem = PhraseGeneratorProblem(data_dir=str(data_dir))
    examples = list(problem.generate_samples(str(data_dir), None, None))
    assert len(examples) == 4
    assert examples[0] == {"inputs": "phrase 1", "targets": "paraphrase 1"}
    assert examples[1] == {"inputs": "paraphrase 1", "targets": "phrase 1"}
    assert examples[2] == {"inputs": "phrase 2", "targets": "paraphrase 2a"}
    assert examples[3] == {"inputs": "phrase 2", "targets": "paraphrase 2b"}
    #permutations will handle the reverse cases as well.

def test_combine_data(clean_test_env):
    file1 = clean_test_env / "file1.tsv"
    file2 = clean_test_env / "file2.tsv"
    output_file = clean_test_env / "combined.tsv"

    with open(file1, "w", encoding="utf-8") as f:
        f.write("line 1\n")
        f.write("line 2\n")

    with open(file2, "w", encoding="utf-8") as f:
        f.write("line 2\n")  # Duplicate
        f.write("line 3\n")

    combine_data(file1, file2, output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert sorted(lines) == ["line 1", "line 2", "line 3"]  # Check content and order

def test_combine_data_one_file_missing(clean_test_env, caplog):
    file1 = clean_test_env / "file1.tsv"
    file2 = clean_test_env / "file2.tsv"  # This file will be missing
    output_file = clean_test_env / "combined.tsv"

    with open(file1, "w", encoding="utf-8") as f:
        f.write("line 1\n")

    combine_data(file1, file2, output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert lines == ["line 1"]
    assert "File does not exist" in caplog.text

def test_postprocess_discriminator_output(clean_test_env):
    labels_file = clean_test_env / "labels.txt"
    generated_file = clean_test_env / "generated.tsv"
    output_file = clean_test_env / "output.tsv"

    with open(labels_file, "w", encoding="utf-8") as f:
        f.write("human_phrase\n")
        f.write("not_human_phrase\n")
        f.write("human_phrase\n")

    with open(generated_file, "w", encoding="utf-8") as f:
        f.write("phrase 1\tparaphrase 1\n")
        f.write("phrase 2\tparaphrase 2\n")
        f.write("phrase 3\tparaphrase 3\n")

    postprocess_discriminator_output(labels_file, generated_file, output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    assert lines == ["phrase 1\tparaphrase 1", "phrase 3\tparaphrase 3"]



def test_generate_mock_paraphrases(clean_test_env):
    output_dir = clean_test_env / "mock_data"
    generate_mock_paraphrases(output_dir, n_repeat_lines=2)

    assert (output_dir / "paraphrases_selected.tsv").exists()
    assert (output_dir / "paraphrases_generated.tsv").exists()

    with open(output_dir / "paraphrases_selected.tsv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 4  # 2 lines repeated twice
    assert lines[0] == "a human phrase\ta human paraphrase\thuman paraphrase\n"

    with open(output_dir / "paraphrases_generated.tsv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 4
    assert lines[0] == "a human phrase\ta machine phrase\tmachine phrase\n"

def test_run_shell_command():
    # Test successful command
    result = run_shell_command("echo hello", capture_output=True)
    assert result.stdout.strip() == "hello"

    # Test command with error
    with pytest.raises(subprocess.CalledProcessError):
        run_shell_command("false")  # 'false' command always returns a non-zero exit code

    # Test command with working directory
    result = run_shell_command("pwd", cwd="/", capture_output=True)
    assert result.stdout.strip() == "/"

    # Test with shell=True (for complex commands; use with caution)
    result = run_shell_command("echo $HOME", capture_output=True, shell=True)
    assert result.stdout.strip() == os.environ['HOME'] #Shell expands the env variable.

def test_ensure_directory(clean_test_env):
     new_dir = clean_test_env / "new_dir" / "subdir"
     ensure_directory(new_dir)
     assert new_dir.is_dir()

     #Test idempotency
     ensure_directory(new_dir)
     assert new_dir.is_dir()  # Should not raise an error if it exists

def test_data_preparer_missing_input_dir():
    with pytest.raises(FileNotFoundError):
        preparer = DataPreparer("missing_dir", "output_dir")
        preparer.prepare()