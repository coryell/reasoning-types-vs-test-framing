"""API keys are loaded from a dotenv file that lives outside the repository.

Default location: ``~/.config/d10/env``. Override with the ``D10_ENV`` environment variable.
Nothing in this repo should ever contain a key.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_ENV_PATH = Path.home() / ".config" / "d10" / "env"


def env_path() -> Path:
    return Path(os.environ.get("D10_ENV", DEFAULT_ENV_PATH)).expanduser()


def load_env() -> Path:
    """Load the dotenv file (without overriding variables already set) and return its path."""
    p = env_path()
    if p.exists():
        load_dotenv(p, override=False)
    return p


def require(key: str) -> str:
    """Return the value of ``key`` from the environment, loading the dotenv file first."""
    p = load_env()
    value = os.environ.get(key, "")
    if not value:
        raise RuntimeError(f"{key} is not set; expected it in {p} (or set D10_ENV)")
    return value
