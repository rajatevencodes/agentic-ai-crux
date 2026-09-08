"""Load the environment and quiet notebook warnings and library logs."""

import logging
import warnings
from pathlib import Path

from dotenv import load_dotenv

_QUIET_LOGGERS = ("", "httpx", "openai")


def configure_notebook() -> Path:
    """Load the project environment and keep notebook output focused.

    Call: env_path = configure_notebook()  (no arguments)
    Returns a Path, not a dict. For JSON, use {"env_path": str(env_path)}.

    Example JSON view, with a placeholder project path:
    ```json
    {"env_path": "<project>/.env"}
    ```
    The file must exist. Its values are loaded into the environment, not printed.
    """
    warnings.filterwarnings("ignore")

    for logger_name in _QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"No .env file found at {env_path}")

    load_dotenv(env_path)
    return env_path
