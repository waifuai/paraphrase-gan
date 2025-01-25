# GAN Paraphrase Generation Framework

A Generative Adversarial Network (GAN) system for iterative paraphrase generation, 
where a generator model creates synthetic paraphrases and a discriminator model 
learns to distinguish human-generated from machine-generated text.

## Prerequisites

- Python 3.7+
- [Trax](https://github.com/google/trax) framework
- Bash shell
- pytest (for testing)

## Installation

1. Clone repository:

2. Install dependencies:
```bash
pip install -r requirements.txt  # Create this file with your dependencies
```

## Directory Structure

```
.
├── config/            # Configuration files
│   └── paths.sh       # Environment paths
├── scripts/           # Main workflow scripts
│   ├── 0_gan_prep/    # Initial model preparation
│   └── 1_gan_loop/    # Continuous training loop
├── src/
│   ├── discriminator/ # Discrimination components
│   └── generator/     # Generation components
├── data/              # Training data (created at runtime)
├── models/            # Trained models (created at runtime)
└── tests/             # Test cases
```

## Usage

### Initial Setup
Create directory structure:
```bash
source config/paths.sh
```

### Full Pipeline Execution
Run the complete GAN workflow:
```bash
./run.sh
```

### Key Components

1. **GAN Preparation** (`scripts/0_gan_prep`):
```bash
Initializes generator and discriminator models with base training data

bash scripts/0_gan_prep/0_gan_prep.sh
```

2. **GAN Training Loop** (`scripts/1_gan_loop`):
```bash
Continuous improvement cycle:
1. Generator creates new paraphrases
2. Discriminator filters best results
3. Both models update with new data

bash scripts/1_gan_loop/1_gan_loop.sh
```

## Testing

Run validation tests:
```bash
pytest tests/
```

### Component Tests
```bash
# Test discriminator training
./src/discriminator/test_discriminator_train_prep.sh
./src/discriminator/test_discriminator_train_do.sh

# Test generator training 
./src/generator/test_generator_train_prep.sh
./src/generator/test_generator_train.sh
```

## Customization

### Modify Training Parameters
Adjust model settings in:
```
src/generator/phrase_generator/trainer/problem.py
src/discriminator/phrase_discriminator/trainer/problem.py
```

### Configure Paths
Update environment paths in:
```bash
config/paths.sh
```

### Change Data Sources
Modify input files:
```
scripts/0_gan_prep/data/input/paraphrases.tsv
scripts/1_gan_loop/data/input/paraphrases.tsv
```