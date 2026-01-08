# service/env.py: loader that makes it easy for Python code to use .env.local
from __future__ import annotations

from pathlib import Path

# First time: downloads and extracts the dataset into the cache
def load_repo_dotenv(filename: str = ".env.local") -> bool:
    """Load an env file from the repo root (or any parent of CWD).

    - Intended for local scripts/tests: keep secrets in `.env` (gitignored).
    - Safe no-op if `python-dotenv` is not installed.

    Returns True if a file was found and load was attempted.
    """

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return False

    cwd = Path.cwd()
    for directory in [cwd, *cwd.parents]:
        env_path = directory / filename
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return True

    return False
