import ast
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from collivind.models.entity import EntityCreate, EntityNode, EntityType
from collivind.models.memory import MemoryCategory, MemoryCreate, MemoryNode, MemorySource
from collivind.models.relationship import RelationshipCreate, RelationshipEdge
from collivind.storage.interfaces import GraphStore

logger = logging.getLogger(__name__)


class SqliteGraphStore(GraphStore):
    """Graph storage using SQLite with adjacency tables. No Docker needed."""

    def __init__(self, data_dir: str):
        db_path = Path(data_dir).expanduser() / "collivind_graph.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def initialize(self) -> None:
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                summary TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                valid_from TEXT,
                valid_to TEXT,
                project_id TEXT DEFAULT 'default',
                session_id TEXT,
                user_id TEXT DEFAULT 'local',
                source TEXT DEFAULT 'manual',
                superseded_by TEXT,
                tags TEXT DEFAULT '[]',
                version INTEGER DEFAULT 1,
                previous_version_id TEXT,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'extraction',
                properties TEXT DEFAULT '{}',
                created_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
        """)
        self.conn.commit()

    def create_memory(self, data: MemoryCreate) -> MemoryNode:
        now = datetime.now(timezone.utc)
        node = MemoryNode(
            id=str(uuid.uuid4()),
            content=data.content,
            summary=data.summary,
            category=data.category,
            confidence=data.confidence,
            valid_from=now,
            project_id=data.project_id,
            session_id=data.session_id,
            user_id=data.user_id,
            source=data.source,
            tags=data.tags,
            created_at=now,
            updated_at=now
        )
        self.conn.execute(
            """INSERT INTO memories (id, content, summary, category, confidence,
               valid_from, valid_to, project_id, session_id, user_id, source,
               superseded_by, tags, version, previous_version_id,
               created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (node.id, node.content, node.summary, node.category.value,
             node.confidence, node.valid_from.isoformat(), None,
             node.project_id, node.session_id, node.user_id,
             node.source.value, None, json.dumps(node.tags),
             node.version, node.previous_version_id,
             node.created_at.isoformat(), node.updated_at.isoformat())
        )
        self.conn.commit()
        return node

    def get_memory(self, id: str) -> Optional[MemoryNode]:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    def update_memory(self, id: str, **updates) -> Optional[MemoryNode]:
        if not updates:
            return self.get_memory(id)
        updates = {k: self._serialize_update_value(k, v) for k, v in updates.items()}
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [id]
        self.conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        self.conn.commit()
        return self.get_memory(id)

    def delete_memory(self, id: str) -> None:
        self.conn.execute("DELETE FROM relationships WHERE source_id = ? OR target_id = ?", (id, id))
        self.conn.execute("DELETE FROM memories WHERE id = ?", (id,))
        self.conn.commit()

    def create_entity(self, data: EntityCreate) -> EntityNode:
        now = datetime.now(timezone.utc)
        entity_id = data.name.lower().replace(" ", "_").replace("-", "_")
        existing = self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if existing:
            return self._row_to_entity(existing)
        node = EntityNode(
            id=entity_id,
            name=data.name,
            type=data.type,
            properties=data.properties or {},
            created_at=now,
            updated_at=now
        )
        self.conn.execute(
            """INSERT INTO entities (id, name, type, properties, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (node.id, node.name, node.type.value, json.dumps(node.properties),
             now.isoformat(), now.isoformat())
        )
        self.conn.commit()
        return node

    def get_entity(self, name: str) -> Optional[EntityNode]:
        entity_id = name.lower().replace(" ", "_").replace("-", "_")
        row = self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if not row:
            return None
        return self._row_to_entity(row)

    def create_relationship(self, data: RelationshipCreate) -> RelationshipEdge:
        now = datetime.now(timezone.utc)
        edge = RelationshipEdge(
            id=str(uuid.uuid4()),
            source_id=data.source_id,
            target_id=data.target_id,
            type=data.type,
            confidence=data.confidence,
            source=data.source,
            properties=data.properties or {},
            created_at=now
        )
        self.conn.execute(
            """INSERT INTO relationships (id, source_id, target_id, type, confidence, source, properties, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (edge.id, edge.source_id, edge.target_id, edge.type.value,
             edge.confidence, edge.source, json.dumps(edge.properties),
             now.isoformat())
        )
        self.conn.commit()
        return edge

    def get_neighbors(
        self, node_id: str, rel_types: List[str],
        direction: str = "OUT", depth: int = 1,
    ) -> List[Dict[str, Any]]:
        results = []
        if direction in ("OUT", "BOTH"):
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE source_id = ?", (node_id,)
            ).fetchall()
            for row in rows:
                if not rel_types or row["type"] in rel_types:
                    results.append({"id": row["target_id"], "rel_type": row["type"], "direction": "OUT"})
        if direction in ("IN", "BOTH"):
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE target_id = ?", (node_id,)
            ).fetchall()
            for row in rows:
                if not rel_types or row["type"] in rel_types:
                    results.append({"id": row["source_id"], "rel_type": row["type"], "direction": "IN"})
        return results

    def find_related_memories(self, entity_name: str, limit: int = 10) -> List[MemoryNode]:
        entity_id = entity_name.lower().replace(" ", "_").replace("-", "_")
        rows = self.conn.execute(
            """SELECT m.* FROM memories m
               JOIN relationships r ON r.source_id = m.id
               WHERE r.target_id = ?
               ORDER BY m.created_at DESC LIMIT ?""",
            (entity_id, limit)
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_timeline(self, project_id: str, entity: Optional[str] = None, limit: int = 50) -> List[MemoryNode]:
        if entity:
            entity_id = entity.lower().replace(" ", "_").replace("-", "_")
            rows = self.conn.execute(
                """SELECT m.* FROM memories m
                   JOIN relationships r ON r.source_id = m.id
                   WHERE m.project_id = ? AND r.target_id = ?
                   ORDER BY m.created_at DESC LIMIT ?""",
                (project_id, entity_id, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, limit)
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def invalidate_memory(self, id: str, superseded_by: str, reason: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE memories SET valid_to = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
            (now, superseded_by, now, id)
        )
        old = self.get_memory(id)
        if old:
            new_mem = self.get_memory(superseded_by)
            if new_mem:
                self.conn.execute(
                    "UPDATE memories SET version = ?, previous_version_id = ? WHERE id = ?",
                    (old.version + 1, id, superseded_by)
                )
        self.conn.commit()

    def get_version_chain(self, memory_id: str) -> List[MemoryNode]:
        current = self.get_memory(memory_id)
        if not current:
            return []
        chain = [current]
        while current.previous_version_id:
            prev = self.get_memory(current.previous_version_id)
            if not prev:
                break
            chain.append(prev)
            current = prev
        chain.reverse()
        start = self.get_memory(memory_id)
        current = start
        while current.superseded_by:
            nxt = self.get_memory(current.superseded_by)
            if not nxt:
                break
            chain.append(nxt)
            current = nxt
        return chain

    def clear_all(self) -> None:
        self.conn.execute("DELETE FROM relationships")
        self.conn.execute("DELETE FROM entities")
        self.conn.execute("DELETE FROM memories")
        self.conn.commit()

    def health_check(self) -> Dict[str, Any]:
        try:
            count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            return {"status": "ok", "message": f"SQLite graph store ({count} memories)"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self) -> None:
        self.conn.close()

    def _row_to_memory(self, row) -> MemoryNode:
        version = row["version"] if "version" in row.keys() else 1
        prev_id = row["previous_version_id"] if "previous_version_id" in row.keys() else None
        return MemoryNode(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            category=MemoryCategory(row["category"]),
            confidence=row["confidence"],
            valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
            valid_to=datetime.fromisoformat(row["valid_to"]) if row["valid_to"] else None,
            project_id=row["project_id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            source=MemorySource(row["source"]) if row["source"] else MemorySource.MANUAL,
            superseded_by=row["superseded_by"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            version=version,
            previous_version_id=prev_id,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
        )

    def _row_to_entity(self, row) -> EntityNode:
        return EntityNode(
            id=row["id"],
            name=row["name"],
            type=EntityType(row["type"]),
            properties=json.loads(row["properties"]) if row["properties"] else {},
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
        )

    def _serialize_update_value(self, key: str, value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        if key != "tags":
            return value
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value))
        if isinstance(value, str):
            try:
                json.loads(value)
                return value
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    parsed = [value]
                if isinstance(parsed, list):
                    return json.dumps(parsed)
                return json.dumps([value])
        return json.dumps(value)
