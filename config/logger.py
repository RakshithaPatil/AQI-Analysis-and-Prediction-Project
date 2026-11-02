"""
config/logger.py
================
Centralised logging setup for the entire project.

Every module should obtain its logger like this:

    from config.logger import get_logger
    logger = get_logger(__name__)

Logs are written to both the console and a rotating file under logs/.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Project root is two levels up from this file
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Log format ───────────────────────────────────────────────────────────────
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a named logger with:
      - A StreamHandler writing to stdout
      - A RotatingFileHandler writing to logs/<log_file>

    Parameters
    ----------
    name : str
        Logger name (use __name__ for the calling module).
    log_file : str
        Log filename inside the logs/ directory.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger was already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (rotating, max 5 MB × 3 backups) ───────────────────────
    file_path = _LOG_DIR / log_file
    file_handler = RotatingFileHandler(
        file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
