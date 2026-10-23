"""Environment variables from the .env file, loaded once, by hand.

No dotenv dependency: our .env file = plain KEY=VALUE lines.
Set values in the SHELL always win (setdefault never overrides).
The file is gitignored; never commit it.
"""
import os
from pathlib import Path


def load_env(path=Path(".env")):
    """Read KEY=VALUE lines into os.environ."""
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())