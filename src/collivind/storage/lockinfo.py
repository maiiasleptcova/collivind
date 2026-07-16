"""Best-effort "who is holding this file" diagnostics for lock errors.

Extracted from the old embedded-Qdrant store; shared by the one-time
qdrant migration and residual SQLite ``database is locked`` errors.
"""

import subprocess
from pathlib import Path
from typing import List


def lock_holders(path: Path) -> List[str]:
    """PIDs with `path` open, via lsof. Best-effort: empty list on any failure."""
    try:
        out = subprocess.run(["lsof", "-t", str(path)], capture_output=True, text=True, timeout=5)
        return [p for p in out.stdout.split() if p.strip()]
    except Exception:
        return []


def lock_message(headline: str, path: Path) -> str:
    """Diagnostic message naming the lock-holding PIDs and concrete recovery steps."""
    pids = lock_holders(path)
    holder = f" (held by PID{'s' if len(pids) > 1 else ''} {', '.join(pids)})" if pids else ""
    recovery = f"kill a stale process (kill {pids[0]})" if pids else f"find the holder (lsof {path})"
    return (
        f"{headline}{holder} — usually another agent session's collivind process. "
        f"Recovery: close the other session, {recovery}, then retry."
    )
