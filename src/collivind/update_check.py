"""Daily PyPI version check with a one-line upgrade notice. Never raises."""

import json
import time
from pathlib import Path

import httpx

from collivind.version import __version__

PYPI_URL = "https://pypi.org/pypi/collivind-memory/json"
CHECK_INTERVAL_SECONDS = 86400


def _state_file() -> Path:
    return Path.home() / ".collivind" / "update_check.json"


def _parse(version: str) -> tuple:
    # ponytail: non-numeric parts (rc/dev suffixes) count as 0; swap in
    # packaging.version if pre-releases ever matter
    parts = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def get_update_notice() -> str | None:
    """One-line upgrade notice when PyPI has a newer version, else None.

    Queries PyPI at most once a day (state cached in ~/.collivind);
    silent on any failure — an update check must never break the CLI.
    """
    try:
        state = {}
        state_file = _state_file()
        if state_file.exists():
            state = json.loads(state_file.read_text())

        now = time.time()
        if now - state.get("checked_at", 0) > CHECK_INTERVAL_SECONDS:
            latest = httpx.get(PYPI_URL, timeout=2.0).json()["info"]["version"]
            state = {"checked_at": now, "latest": latest}
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(state))

        latest = state.get("latest")
        if latest and _parse(latest) > _parse(__version__):
            return (
                f"Collivind {latest} is available (you have {__version__}) — "
                f"upgrade with: pip install -U collivind-memory"
            )
    except Exception:
        pass
    return None
