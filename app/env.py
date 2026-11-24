"""Environment variable loader for PRMPT.

Loads variables from .env into os.environ without overwriting existing environment variables.
Does not require external dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(env_path: Path | str | None = None) -> dict[str, str]:
    """Parse a .env file and populate os.environ for unset variables.

    Returns dictionary of variables parsed from the file.
    """
    path = Path(env_path) if env_path else ENV_PATH
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return loaded

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Handle quoted values
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            elif " #" in val:
                # Strip inline comments for unquoted values
                val = val.split(" #", 1)[0].strip()

            if key:
                loaded[key] = val
                if key not in os.environ:
                    os.environ[key] = val
    return loaded


# Automatically load on import
load_env()
