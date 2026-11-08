"""Capture the current code version from the git working tree.

Returns the full 40-character commit SHA when git is available, or
``None`` when it cannot be determined.  Callers should represent an
unavailable version explicitly rather than substituting a placeholder.
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def current_commit() -> str | None:
    """Return the HEAD commit SHA, or ``None`` if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
        )
        sha = result.stdout.strip()
        if result.returncode == 0 and len(sha) == 40:
            return sha
    except Exception as exc:
        log.debug("Could not read git HEAD: %s", exc)
    return None
