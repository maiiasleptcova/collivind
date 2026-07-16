"""SQLite-backed vector store: WAL blobs + in-process numpy cosine scan.

Multi-process safe: writes go through SQLite (WAL journal + busy_timeout),
and readers detect other processes' commits via ``PRAGMA data_version``,
reloading an in-RAM float32 matrix before every scan that needs it. This
gives strict cross-process read-your-writes — the property the previous
engine (qdrant-client local mode, exclusive directory lock) could not offer.

Semantics mirror qdrant-client's local mode exactly, pinned by tests:

- vectors are L2-normalized float32 at write; the score is the cosine dot
  product, accumulated in float64 like QdrantLocal's scoring pipeline
- the score threshold is inclusive (``score >= threshold`` is kept)
- filters use flat-key MatchValue semantics: the payload value equals the
  filter value, or (for list payload values) contains it. Filter values must
  be str|int|bool, like MatchValue. Dotted key paths are unsupported
  (unused in-tree).

Legacy data in ``<data_dir>/qdrant_data`` is migrated once on first open and
left in place (read, never mutated) so a package rollback still finds it.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from collivind.config import QdrantConfig
from collivind.exceptions import CollivindError
from collivind.storage import lockinfo
from collivind.storage.interfaces import VectorStore

logger = logging.getLogger(__name__)

# mirrors QdrantLocal's zero-vector guard (Rust f32::EPSILON)
EPSILON = 1.1920929e-7

LOCK_HINT = "already accessed by another instance"

_MIGRATION_FLAG = "migrated_qdrant_v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    collection TEXT NOT NULL,
    id         TEXT NOT NULL,
    vector     BLOB NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (collection, id)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# cache of the configured collection: (data_version at load, ids, matrix, payloads)
_Cache = Tuple[int, List[str], np.ndarray, List[Dict[str, Any]]]


def _normalize(vector: Any) -> np.ndarray:
    """L2-normalize to float32 — the same math QdrantLocal runs at query time."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / (norm or EPSILON)


def _validate_filters(filters: Dict[str, Any]) -> None:
    """Old-engine parity: qdrant MatchValue accepts only str|int|bool (R3).

    Anything else raised ValidationError there; silently accepting it here
    made ``{'k': None}`` match every point missing key ``k``.
    """
    for key, value in filters.items():
        if not isinstance(value, (str, int)):  # bool is a subclass of int
            raise CollivindError(
                f"Unsupported filter value for {key!r}: {type(value).__name__} (expected str, int, or bool)"
            )


def _matches(payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Qdrant MatchValue semantics for flat keys: equal, or contained in a list value."""
    for key, expected in filters.items():
        value = payload.get(key)
        if value == expected:
            continue
        if isinstance(value, list) and expected in value:
            continue
        return False
    return True


class SqliteVectorStore(VectorStore):
    """Multi-process-safe vector store on a single SQLite file."""

    def __init__(self, data_dir: str, config: QdrantConfig, dimension: int):
        self.config = config
        self._dimension = dimension
        base = Path(data_dir).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        self._db_path = base / "collivind_vectors.db"
        self._legacy_dir = base / "qdrant_data"
        self.conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout=10000")  # before the WAL switch: every later statement is covered
        self._enable_wal()
        self._ensure_schema()
        self._cache: Optional[_Cache] = None
        self._migrate_legacy_qdrant()

    # --- VectorStore interface ---

    def initialize(self) -> None:
        try:
            self._ensure_schema()
        except Exception as e:
            raise self._wrap("initialization", e)

    def upsert(self, id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        if len(vector) != self._dimension:
            raise CollivindError(f"Vector dimension {len(vector)} does not match store dimension {self._dimension}")
        try:
            self._check_collection_dimension(len(vector))
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO vectors (collection, id, vector, payload) VALUES (?, ?, ?, ?)",
                    (self.config.collection_name, str(id), _normalize(vector).tobytes(), json.dumps(payload)),
                )
            # own commits never tick data_version — invalidate the cache directly
            self._cache = None
        except Exception as e:
            raise self._wrap("upsert", e)

    def search(
        self,
        vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        try:
            if filters:
                _validate_filters(filters)
            _, ids, matrix, payloads = self._load_cache()
            if not ids:
                return []
            # float64 query promotes the whole matmul to float64 accumulation,
            # matching QdrantLocal's pipeline at the 0.92/0.98 dedup boundaries
            # (R4). Storage stays float32; the transient float64 matrix copy
            # costs ~0.5 ms and ~30 MB per search at 10k x 384 — fine here.
            scores = matrix @ _normalize(vector).astype(np.float64)
            results: List[Dict[str, Any]] = []
            for idx in np.argsort(scores)[::-1]:
                score = float(scores[idx])
                if score < threshold:  # inclusive threshold: score == threshold is kept
                    break
                if filters and not _matches(payloads[idx], filters):
                    continue
                results.append({"id": ids[idx], "score": score, "payload": dict(payloads[idx])})
                if len(results) >= limit:
                    break
            return results
        except Exception as e:
            raise self._wrap("search", e)

    def delete(self, id: str) -> None:
        try:
            with self.conn:
                self.conn.execute(
                    "DELETE FROM vectors WHERE collection = ? AND id = ?", (self.config.collection_name, str(id))
                )
            self._cache = None
        except Exception as e:
            raise self._wrap("delete", e)

    def delete_collection(self) -> None:
        try:
            with self.conn:
                self.conn.execute("DELETE FROM vectors WHERE collection = ?", (self.config.collection_name,))
            self._cache = None
        except Exception as e:
            raise self._wrap("delete_collection", e)

    def health_check(self) -> Dict[str, Any]:
        try:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection = ?", (self.config.collection_name,)
            ).fetchone()[0]
            return {"status": "ok", "message": f"SQLite vector store ({count} vectors)"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self) -> None:
        self.conn.close()

    # --- freshness ---

    def _load_cache(self) -> _Cache:
        """Return the collection cache, reloading when another process committed.

        The order is load-bearing: read data_version BEFORE the SELECT, then
        stamp the cache with that pre-read value. A commit landing mid-load can
        only mark the cache older than its contents (harmless extra reload);
        the reverse order would mark stale data fresh.
        """
        version = self.conn.execute("PRAGMA data_version").fetchone()[0]
        cache = self._cache
        if cache is not None and cache[0] == version:
            return cache
        rows = self.conn.execute(
            "SELECT id, vector, payload FROM vectors WHERE collection = ?", (self.config.collection_name,)
        ).fetchall()
        ids = [row[0] for row in rows]
        if rows:
            matrix = np.vstack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        else:
            matrix = np.empty((0, self._dimension), dtype=np.float32)
        payloads = [json.loads(row[2]) for row in rows]
        cache = (version, ids, matrix, payloads)
        self._cache = cache  # atomic swap: concurrent readers never see a torn cache
        return cache

    # --- plumbing ---

    def _enable_wal(self) -> None:
        """Switch to WAL, retrying the first-creation race.

        The rollback->WAL conversion takes an exclusive lock and SQLite does
        NOT consult the busy handler for it, so N processes cold-starting on
        a fresh directory raced into raw 'database is locked' (~1 ms window;
        review finding R1). The losers just retry — the winner's conversion
        is instant and persistent, so one retry normally suffices.
        """
        for attempt in range(10):
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 9:
                    raise self._wrap("open", e) from e
                time.sleep(0.01 * (attempt + 1))

    def _ensure_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def _check_collection_dimension(self, dim: int) -> None:
        """Reject writes that mismatch the collection's established dimension.

        The old engine pinned the dimension at collection creation; without
        this, one accepted mismatched row bricks every search on the
        collection (np.vstack shape mismatch, review finding R2). Rows are
        normalized float32, so the stored blob is 4 bytes per component.
        """
        row = self.conn.execute(
            "SELECT length(vector) FROM vectors WHERE collection = ? LIMIT 1", (self.config.collection_name,)
        ).fetchone()
        if row is not None and row[0] != dim * 4:
            raise CollivindError(
                f"Vector dimension {dim} does not match existing dimension {row[0] // 4} "
                f"of collection '{self.config.collection_name}'"
            )

    def _meta_get(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _wrap(self, operation: str, error: Exception) -> CollivindError:
        if isinstance(error, CollivindError):
            return error  # already a boundary error: don't bury the message
        if isinstance(error, sqlite3.OperationalError) and "locked" in str(error):
            return CollivindError(
                lockinfo.lock_message(f"Vector store {operation} failed ({error}): {self._db_path}", self._db_path)
            )
        return CollivindError(f"Vector store {operation} failed: {error}")

    # --- one-time migration from qdrant-client local mode ---

    def _migrate_legacy_qdrant(self) -> None:
        if self._meta_get(_MIGRATION_FLAG):
            return
        if not self._legacy_dir.exists():
            # fresh install: nothing to migrate, qdrant is never imported
            with self.conn:
                self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, '1')", (_MIGRATION_FLAG,))
            return
        if not self._begin_exclusive_or_wait():
            return  # another process finished the migration while we waited
        try:
            if self._meta_get(_MIGRATION_FLAG):  # lost the race, winner already committed
                self.conn.execute("ROLLBACK")
                return
            self._copy_legacy_points()
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, '1')", (_MIGRATION_FLAG,))
            self.conn.execute("COMMIT")
            logger.info(f"Migrated legacy embedded Qdrant data from {self._legacy_dir}")
        except BaseException as e:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            if isinstance(e, Exception) and not isinstance(e, CollivindError):
                # corrupt/unreadable legacy data (M1/R5): fail loud but guided,
                # never a raw qdrant-internal or JSONDecodeError
                raise CollivindError(
                    f"Cannot migrate legacy embedded Qdrant data at {self._legacy_dir}: {e}. "
                    f"The legacy data was not modified. To recover, move the directory aside — "
                    f"e.g. `mv {self._legacy_dir} {self._legacy_dir}.corrupt` — and rerun: the store "
                    f"starts fresh and the original data stays preserved for inspection."
                ) from e
            raise

    def _begin_exclusive_or_wait(self) -> bool:
        """Serialize racing processes; False when the winner set the flag meanwhile."""
        for _ in range(3):
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                return True
            except sqlite3.OperationalError:
                time.sleep(2)  # another process is likely mid-migration
                if self._meta_get(_MIGRATION_FLAG):
                    return False
        raise CollivindError(
            lockinfo.lock_message(
                f"Vector store at {self._db_path} stayed locked while waiting for legacy-data migration",
                self._db_path,
            )
        )

    def _copy_legacy_points(self) -> None:
        """Copy every collection (not just the configured one) out of qdrant_data."""
        client = self._open_legacy_qdrant()
        try:
            for collection in client.get_collections().collections:
                offset = None
                while True:
                    points, offset = client.scroll(
                        collection_name=collection.name, limit=256, offset=offset, with_payload=True, with_vectors=True
                    )
                    for point in points:
                        self.conn.execute(
                            "INSERT OR REPLACE INTO vectors (collection, id, vector, payload) VALUES (?, ?, ?, ?)",
                            (
                                collection.name,
                                str(point.id),
                                _normalize(point.vector).tobytes(),
                                json.dumps(point.payload or {}),
                            ),
                        )
                    if offset is None:
                        break
        finally:
            client.close()

    def _open_legacy_qdrant(self):
        from qdrant_client import QdrantClient

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                return QdrantClient(path=str(self._legacy_dir))
            except Exception as e:
                if LOCK_HINT not in str(e):
                    raise
                last_error = e
                time.sleep(0.5 * (attempt + 1))
        raise CollivindError(
            lockinfo.lock_message(
                f"Cannot migrate legacy embedded Qdrant data at {self._legacy_dir}: it is locked by a "
                f"collivind process still running a pre-upgrade version",
                self._legacy_dir / ".lock",
            )
        ) from last_error
