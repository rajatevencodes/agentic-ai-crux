"""Reusable setup shared by Agentic AI Crux notebooks."""

import logging
import warnings
from pathlib import Path

from dotenv import load_dotenv


def configure_notebook() -> Path:
    """Load the project environment and keep notebook output focused."""
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    for logger_name in ("httpx", "openai"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    return env_path
