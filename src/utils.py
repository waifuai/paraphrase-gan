import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

# Import config relative to src directory
from .config import CONFIG

# --- Helper Functions ---

def setup_logger(log_dir: Path, config: Dict[str, Any]) -> logging.Logger:
    """Sets up a logger with console and file handlers based on config."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / config['filenames']['log_file']
    logger = logging.getLogger(config['logging']['logger_name'])
    logger.setLevel(logging.DEBUG) # Set root logger level to lowest

    # Prevent adding handlers multiple times if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(config['logging']['console_level'])
    formatter_ch = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(formatter_ch)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(config['logging']['file_level'])
    formatter_fh = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(message)s"
    )
    fh.setFormatter(formatter_fh)
    logger.addHandler(fh)

    return logger

# Initialize logger using the config from config.py
# This makes the logger available for import from this module
logger = setup_logger(CONFIG['paths']['logs_dir'], CONFIG)


def run_shell_command(
    command: list[str], # Use list for better security than shell=True
    cwd: str = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """
    Runs a shell command with optional working directory and output capture.
    Provides more consistent error handling. Avoids shell=True.
    """
    try:
        logger.debug(f"Running command: {' '.join(command)} in {cwd or os.getcwd()}")
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=True,
            shell=False, # Avoid shell=True for security
        )
        if capture_output:
             logger.debug(f"Command stdout:\n{result.stdout}")
             logger.debug(f"Command stderr:\n{result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(e.cmd)}")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr}")
        raise  # Re-raise the exception to stop execution
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        raise


def ensure_directory(path: Path):
    """Creates a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")