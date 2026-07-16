"""Embedded vector store for no-Docker deployments.

Historically this wrapped qdrant-client's local mode, which holds an
exclusive lock on its storage directory — only ONE collivind process
(MCP server, CLI verb, or hook) could run at a time. The embedded store
is now SQLite (WAL) + a numpy cosine scan (see
:mod:`collivind.storage.vector_sqlite`), safe for any number of concurrent
processes. Pre-existing ``qdrant_data`` is migrated automatically on first
open and left in place.

The class name and constructor signature are kept so the factory, the
benchmark harness, and existing test patches keep working unchanged.
"""

from collivind.storage.vector_sqlite import SqliteVectorStore


class EmbeddedQdrantStore(SqliteVectorStore):
    """SQLite-backed embedded vector store (formerly qdrant-client local mode)."""
