import ast
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j import exceptions as neo4j_exceptions

from collivind.config import Neo4jConfig
from collivind.exceptions import CollivindError
from collivind.models import EntityCreate, EntityNode, MemoryCreate, MemoryNode, RelationshipCreate, RelationshipEdge
from collivind.models.entity import EntityType
from collivind.storage.interfaces import GraphStore

logger = logging.getLogger(__name__)


def _retry(fn, max_retries=3, base_delay=0.5):
    """Retry with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    raise last_error


class Neo4jGraphStore(GraphStore):
    def __init__(self, config: Neo4jConfig):
        self.config = config
        try:
            self.driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
        except Exception as e:
            raise CollivindError(f"Failed to connect to Neo4j: {e}")

    def initialize(self) -> None:
        """Create constraints and indexes."""
        queries = [
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
        ]
        with self.driver.session(database=self.config.database) as session:
            for q in queries:
                try:
                    session.run(q)
                except neo4j_exceptions.ClientError:
                    # Ignore if constraint already exists or not supported in this Neo4j version exactly
                    pass

    def clear_all(self) -> None:
        """Delete all nodes and relationships from the database."""
        with self.driver.session(database=self.config.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_memory(self, data: MemoryCreate) -> MemoryNode:
        memory = MemoryNode(
            content=data.content,
            summary=data.summary,
            category=data.category,
            project_id=data.project_id,
            session_id=data.session_id,
            user_id=data.user_id,
            source=data.source,
            confidence=data.confidence,
            tags=data.tags,
        )

        query = """
        CREATE (m:Memory $props)
        RETURN m
        """

        props = memory.to_dict()

        def _do_create():
            with self.driver.session(database=self.config.database) as session:
                session.run(query, props=props)

        try:
            _retry(_do_create)
        except Exception as e:
            raise CollivindError(f"Failed to create memory in Neo4j after retries: {e}")

        return memory

    def get_memory(self, id: str) -> Optional[MemoryNode]:
        query = "MATCH (m:Memory {id: $id}) RETURN m"

        def _do_get():
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, id=id)
                record = result.single()
                if not record:
                    return None
                node = record["m"]

                # Reconstruct MemoryNode
                return MemoryNode(
                    id=node["id"],
                    content=node["content"],
                    summary=node["summary"],
                    category=node["category"],
                    confidence=node.get("confidence", 1.0),
                    valid_from=datetime.fromisoformat(node["valid_from"]) if node.get("valid_from") else None,
                    valid_to=datetime.fromisoformat(node["valid_to"]) if node.get("valid_to") else None,
                    project_id=node.get("project_id", "default"),
                    session_id=node.get("session_id"),
                    user_id=node.get("user_id", "local"),
                    source=node.get("source", "manual"),
                    superseded_by=node.get("superseded_by"),
                    tags=node.get("tags", []),
                    version=node.get("version", 1),
                    previous_version_id=node.get("previous_version_id"),
                    created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else None,
                    updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else None,
                )

        try:
            return _retry(_do_get)
        except Exception as e:
            raise CollivindError(f"Failed to get memory from Neo4j after retries: {e}")

    def update_memory(self, id: str, **updates) -> Optional[MemoryNode]:
        if not updates:
            return self.get_memory(id)
        updates = {k: self._normalize_update_value(k, v) for k, v in updates.items()}
        set_clauses = ", ".join([f"m.{k} = ${k}" for k in updates.keys()])
        query = f"MATCH (m:Memory {{id: $id}}) SET {set_clauses} RETURN m"

        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, id=id, **updates)
            record = result.single()
            if not record:
                return None
            return self.get_memory(id)

    def delete_memory(self, id: str) -> None:
        query = "MATCH (m:Memory {id: $id}) DETACH DELETE m"
        with self.driver.session(database=self.config.database) as session:
            session.run(query, id=id)

    def create_entity(self, data: EntityCreate) -> EntityNode:
        entity = EntityNode(name=data.name, type=data.type, properties=data.properties)

        query = """
        MERGE (e:Entity {id: $id})
        ON CREATE SET e.name = $name, e.type = $type, e.properties = $props,
            e.created_at = $created_at, e.updated_at = $updated_at
        RETURN e
        """
        with self.driver.session(database=self.config.database) as session:
            session.run(
                query,
                id=entity.id,
                name=entity.name,
                type=entity.type.value,
                props=json.dumps(entity.properties),
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat(),
            )
        return entity

    def get_entity(self, name: str) -> Optional[EntityNode]:
        entity_id = name.lower().replace(" ", "_").replace("-", "_")
        query = "MATCH (e:Entity {id: $id}) RETURN e"
        with self.driver.session(database=self.config.database) as session:
            record = session.run(query, id=entity_id).single()
            if not record:
                return None
            node = record["e"]

        entity_type = node.get("type", EntityType.CONCEPT.value)
        try:
            entity_type = EntityType(entity_type)
        except (TypeError, ValueError):
            entity_type = EntityType.CONCEPT

        return EntityNode(
            id=node["id"],
            name=node.get("name", name),
            type=entity_type,
            properties=self._parse_properties(node.get("properties", {})),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else None,
            updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else None,
        )

    def create_relationship(self, data: RelationshipCreate) -> RelationshipEdge:
        query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        MERGE (source)-[r:{data.type.value}]->(target)
        ON CREATE SET r.confidence = $confidence, r.source = $source
        RETURN r
        """
        with self.driver.session(database=self.config.database) as session:
            session.run(
                query,
                source_id=data.source_id,
                target_id=data.target_id,
                confidence=data.confidence,
                source=data.source,
            )

        return RelationshipEdge(
            source_id=data.source_id,
            target_id=data.target_id,
            type=data.type,
            confidence=data.confidence,
            source=data.source,
        )

    def get_neighbors(
        self,
        node_id: str,
        rel_types: List[str],
        direction: str = "OUT",
        depth: int = 1,
    ) -> List[Dict[str, Any]]:
        type_filter = "|".join(rel_types) if rel_types else ""
        rel_pattern = f"-[r:{type_filter}*1..{depth}]-"
        if direction == "OUT":
            rel_pattern = f"-[r:{type_filter}*1..{depth}]->"
        elif direction == "IN":
            rel_pattern = f"<-[r:{type_filter}*1..{depth}]-"

        query = f"""
        MATCH (n {{id: $node_id}}){rel_pattern}(m)
        RETURN m, r
        """
        results = []
        with self.driver.session(database=self.config.database) as session:
            res = session.run(query, node_id=node_id)
            for record in res:
                node = dict(record["m"])
                results.append(
                    {
                        "id": node.get("id"),
                        "name": node.get("name"),
                        "node": node,
                        "relationships": [dict(rel) for rel in record["r"]],
                    }
                )
        return results

    def find_related_memories(self, entity_name: str, limit: int = 10) -> List[MemoryNode]:
        entity_id = entity_name.lower().replace(" ", "_").replace("-", "_")
        query = """
        MATCH (m:Memory)-[:ABOUT|MENTIONS]->(e:Entity {id: $entity_id})
        RETURN m LIMIT $limit
        """
        memories = []
        with self.driver.session(database=self.config.database) as session:
            res = session.run(query, entity_id=entity_id, limit=limit)
            for record in res:
                node = record["m"]
                memories.append(self.get_memory(node["id"]))  # Simplification: refetch to construct node
        return [m for m in memories if m is not None]

    def get_timeline(self, project_id: str, entity: Optional[str] = None, limit: int = 50) -> List[MemoryNode]:
        if entity:
            entity_id = entity.lower().replace(" ", "_").replace("-", "_")
            query = """
            MATCH (m:Memory {project_id: $project_id})-[:ABOUT|MENTIONS]->(e:Entity {id: $entity_id})
            RETURN m.id as id ORDER BY m.created_at DESC LIMIT $limit
            """
            params = {"project_id": project_id, "entity_id": entity_id, "limit": limit}
        else:
            query = """
            MATCH (m:Memory {project_id: $project_id})
            RETURN m.id as id ORDER BY m.created_at DESC LIMIT $limit
            """
            params = {"project_id": project_id, "limit": limit}

        memories = []
        with self.driver.session(database=self.config.database) as session:
            res = session.run(query, **params)
            for record in res:
                memories.append(self.get_memory(record["id"]))
        return [m for m in memories if m is not None]

    def invalidate_memory(self, id: str, superseded_by: str, reason: str) -> None:
        now_str = datetime.now(timezone.utc).isoformat()

        # Set valid_to and superseded_by property
        self.update_memory(id, valid_to=now_str, superseded_by=superseded_by)

        old = self.get_memory(id)
        new_mem = self.get_memory(superseded_by)
        if old and new_mem:
            self.update_memory(superseded_by, version=old.version + 1, previous_version_id=id)

        # Create relation
        query = """
        MATCH (old:Memory {id: $id})
        MATCH (new:Memory {id: $superseded_by})
        MERGE (new)-[r:SUPERSEDES]->(old)
        SET r.reason = $reason, r.created_at = $now_str
        """
        with self.driver.session(database=self.config.database) as session:
            session.run(query, id=id, superseded_by=superseded_by, reason=reason, now_str=now_str)

    def get_version_chain(self, memory_id: str) -> List[MemoryNode]:
        chain = []
        current = self.get_memory(memory_id)
        if not current:
            return chain
        while current.previous_version_id:
            prev = self.get_memory(current.previous_version_id)
            if not prev:
                break
            chain.append(prev)
            current = prev
        chain.reverse()
        current = self.get_memory(memory_id)
        chain.append(current)
        while current.superseded_by:
            nxt = self.get_memory(current.superseded_by)
            if not nxt:
                break
            chain.append(nxt)
            current = nxt
        return chain

    def health_check(self) -> Dict[str, Any]:
        try:
            with self.driver.session(database=self.config.database) as session:
                res = session.run("RETURN 1 as ok")
                if res.single()["ok"] == 1:
                    return {"status": "ok"}
            return {"status": "error", "message": "Query failed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self) -> None:
        self.driver.close()

    def _normalize_update_value(self, key: str, value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        if key == "tags" and isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    parsed = [value]
            return parsed if isinstance(parsed, list) else [value]
        return value

    def _parse_properties(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str) or not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}
        return parsed if isinstance(parsed, dict) else {}
