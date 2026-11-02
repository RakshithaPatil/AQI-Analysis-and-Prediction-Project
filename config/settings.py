"""
config/settings.py
==================
Loads configuration from config.yaml and environment variables (.env).

Usage:
    from config.settings import settings, get_db_url
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file ──────────────────────────────────────────────────────────
# Looks for .env in the project root (one level above this file's directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)
else:
    # .env not found – warn but continue (Docker injects env vars directly)
    import warnings
    warnings.warn(
        f".env file not found at {_ENV_FILE}. "
        "Falling back to system environment variables.",
        stacklevel=2,
    )

# ── Load config.yaml ────────────────────────────────────────────────────────
_CONFIG_FILE = _PROJECT_ROOT / "config" / "config.yaml"

def _load_yaml(path: Path) -> dict:
    """Read and return the YAML config as a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# The global settings dict – imported by all modules
settings: dict = _load_yaml(_CONFIG_FILE)


# ── Convenience helpers ─────────────────────────────────────────────────────

def get_db_url() -> str:
    """Return the PostgreSQL connection URL from the environment."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    # Build from individual parts if DATABASE_URL is not set
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "aqi_db")
    user = os.getenv("DB_USER", "aqi_user")
    password = os.getenv("DB_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def get_data_path(subfolder: str = "") -> Path:
    """Return an absolute path inside the data/ directory."""
    base = _PROJECT_ROOT / "data"
    if subfolder:
        return base / subfolder
    return base


def get_openaq_api_key() -> str:
    """Return the OpenAQ API key from the environment."""
    return os.getenv("OPENAQ_API_KEY", "")


def get_mlflow_tracking_uri() -> str:
    """Return the MLflow tracking URI."""
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
