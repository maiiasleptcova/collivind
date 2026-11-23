"""Guards for running the UI inside a container.

Embedded mode keeps its stores in SQLite files with WAL journaling, which
coordinates writers through a shared-memory file (`-wal`/`-shm`) and POSIX
locks. Both only work when every process sees the files through the same
kernel. A Linux host's bind mount does; Docker Desktop and Colima on macOS
share the data dir across a VM boundary (virtiofs, gRPC FUSE, sshfs, 9p),
where the host CLI and the container each keep their own lock state and can
corrupt the database (sqlite.org/wal.html, "WAL does not work over a network
filesystem").
"""

import os
from pathlib import Path
from typing import Optional

SHARED_FS = ("virtiofs", "9p", "fuse", "osxfs", "nfs", "cifs", "smb", "fakeowner", "grpcfuse")
ALLOW_ENV = "COLLIVIND_WEB_ALLOW_EMBEDDED"


def in_container() -> bool:
    return os.environ.get("COLLIVIND_IN_CONTAINER") == "1"


def mount_fstype(path: Path, mounts: str) -> Optional[str]:
    """Filesystem type of the longest mount point containing `path`, from /proc/mounts text."""
    target = os.path.abspath(path)
    best, best_type = "", None
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        point = fields[1].replace("\\040", " ")
        if (target == point or target.startswith(point.rstrip("/") + "/")) and len(point) >= len(best):
            best, best_type = point, fields[2]
    return best_type


def embedded_refusal(data_dir: Path, mounts_file: Path = Path("/proc/mounts")) -> Optional[str]:
    """Why embedded mode must not run here, or None when it is safe."""
    if os.environ.get(ALLOW_ENV) == "1":
        return None
    try:
        fstype = mount_fstype(data_dir, mounts_file.read_text())
    except OSError:
        fstype = None
    if fstype is None:
        return f"cannot tell what filesystem {data_dir} is on, so SQLite locking cannot be trusted"
    if any(fstype.startswith(kind) or kind in fstype for kind in SHARED_FS):
        return (
            f"{data_dir} is a {fstype} mount shared across a VM boundary (Docker Desktop or Colima on macOS). "
            "SQLite's WAL locks do not reach across it, so this container and the collivind CLI on the host "
            "could corrupt the embedded store. Run `collivind web` on the host instead, or use docker mode. "
            f"If you know this mount shares the host's kernel, set {ALLOW_ENV}=1."
        )
    return None
