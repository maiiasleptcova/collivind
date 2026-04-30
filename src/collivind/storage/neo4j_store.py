from typing import List, Dict, Any, Optional
from datetime import datetime
from neo4j import GraphDatabase, exceptions as neo4j_exceptions

from collivind.config import Neo4jConfig
from collivind.storage.interfaces import GraphStore
from collivind.models import (
    MemoryNode, MemoryCreate,
    EntityNode, EntityCreate,
    RelationshipEdge, RelationshipCreate
)
from collivind.exceptions import CollivindError

class Neo4jGraphStore(GraphStore):
    def __init__(self, config: Neo4jConfig):
        self.config = config
        try:
            self.driver = GraphDatabase.driver(
                config.uri,
                auth=(config.user, config.password)
            )
        except Exception as e:
            raise CollivindError(f"Failed to connect to Neo4j: {e}")

    def initialize(self) -> None:
        """Create constraints and indexes."""
        queries = [
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE"
        ]
        with self.driver.session(database=self.config.database) as session:
            for q in queries:
                try:
                    session.run(q)
                except neo4j_exceptions.ClientError as e:
                    # Ignore if constraint already exists or not supported in this Neo4j version exactly
                    pass

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
            tags=data.tags
        )
        
        query = """
        CREATE (m:Memory $props)
        RETURN m
        """
        
        props = memory.to_dict()
        
        with self.driver.session(database=self.config.database) as session:
            try:
                session.run(query, props=props)
            except Exception as e:
                raise CollivindError(f"Failed to create memory in Neo4j: {e}")
                
        return memory

    def get_memory(self, id: str) -> Optional[MemoryNode]:
        query = "MATCH (m:Memory {id: $id}) RETURN m"
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
                created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else None,
                updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else None
            )

    def update_memory(self, id: str, **updates) -> Optional[MemoryNode]:
        if not updates:
            return self.get_memory(id)
            
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
        entity = EntityNode(
            name=data.name,
            type=data.type,
            properties=data.properties
        )
        
        query = """
        MERGE (e:Entity {id: $id})
        ON CREATE SET e.name = $name, e.type = $type, e.properties = $props, e.created_at = $created_at, e.updated_at = $updated_at
        RETURN e
        """
        with self.driver.session(database=self.config.database) as session:
            session.run(query, 
                id=entity.id, 
                name=entity.name, 
                type=entity.type.value, 
                props=str(entity.properties), # simplified, production would serialize nicely
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat()
            )
        return entity

    def create_relationship(self, data: RelationshipCreate) -> RelationshipEdge:
        query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        MERGE (source)-[r:{data.type.value}]->(target)
        ON CREATE SET r.confidence = $confidence, r.source = $source
        RETURN r
        """
        with self.driver.session(database=self.config.database) as session:
            session.run(query, 
                source_id=data.source_id, 
                target_id=data.target_id, 
                confidence=data.confidence, 
                source=data.source
            )
        
        return RelationshipEdge(
            source_id=data.source_id,
            target_id=data.target_id,
            type=data.type,
            confidence=data.confidence,
            source=data.source
        )

    def get_neighbors(self, node_id: str, rel_types: List[str], direction: str = "OUT", depth: int = 1) -> List[Dict[str, Any]]:
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
                results.append({
                    "node": dict(record["m"]),
                    "relationships": [dict(rel) for rel in record["r"]]
                })
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
                memories.append(self.get_memory(node["id"])) # Simplification: refetch to construct node
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
        
        # Create relation
        query = """
        MATCH (old:Memory {id: $id})
        MATCH (new:Memory {id: $superseded_by})
        MERGE (new)-[r:SUPERSEDES]->(old)
        SET r.reason = $reason, r.created_at = $now_str
        """
        with self.driver.session(database=self.config.database) as session:
            session.run(query, id=id, superseded_by=superseded_by, reason=reason, now_str=now_str)

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
