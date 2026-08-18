"""
core_utils.py

Pure utility helpers for data processing, validation, and IO.
This module contains NO ML logic and NO domain knowledge.
"""

import logging
from pathlib import Path

import pandas as pd

from .pipeline_errors import SchemaValidationFault


def get_logger(name: str, log_dir: str | None = "logs") -> logging.Logger:
    """Configures and returns a logger with console and file handlers."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_dir:
        ensure_dir(log_dir)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def ensure_dir(dir_path: str) -> str:
    """Ensures a directory exists, creating it if necessary."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

def require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Validates that all required columns exist in the DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise SchemaValidationFault(f"Missing required columns: {missing}")
