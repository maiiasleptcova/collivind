from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple

from collivind.models.entity import EntityCreate, EntityNode
from collivind.models.memory import MemoryCreate, MemoryNode
from collivind.models.relationship import RelationshipCreate, RelationshipEdge


class VectorStore(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store (e.g. create collections)."""
        pass

    @abstractmethod
    def upsert(self, id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        """Upsert a vector with its payload."""
        pass

    @abstractmethod
    def search(
        self,
        vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a vector by id."""
        pass

    @abstractmethod
    def iter_all(self) -> Iterator[Tuple[str, List[float], Dict[str, Any]]]:
        """Yield every (id, vector, payload) in the configured collection.
        `upsert` is the id-preserving restore."""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the configured collection (used by `collivind reset`)."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection."""
        pass


class GraphStore(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the graph store (e.g. constraints/indexes)."""
        pass

    @abstractmethod
    def create_memory(self, data: MemoryCreate) -> MemoryNode:
        """Create a new memory node."""
        pass

    @abstractmethod
    def get_memory(self, id: str) -> Optional[MemoryNode]:
        """Get a memory node by id."""
        pass

    @abstractmethod
    def update_memory(self, id: str, **updates) -> Optional[MemoryNode]:
        """Update a memory node."""
        pass

    @abstractmethod
    def delete_memory(self, id: str) -> None:
        """Delete a memory node."""
        pass

    @abstractmethod
    def create_entity(self, data: EntityCreate) -> EntityNode:
        """Create a new entity node or return existing."""
        pass

    @abstractmethod
    def get_entity(self, name: str) -> Optional[EntityNode]:
        """Get an entity by display name or normalized id."""
        pass

    @abstractmethod
    def create_relationship(self, data: RelationshipCreate) -> RelationshipEdge:
        """Create a relationship edge."""
        pass

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        rel_types: List[str],
        direction: str = "OUT",
        depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes in the graph."""
        pass

    @abstractmethod
    def find_related_memories(self, entity_name: str, limit: int = 10) -> List[MemoryNode]:
        """Find memories related to an entity."""
        pass

    def get_neighbors_batch(
        self, node_ids: List[str], rel_types: List[str], direction: str = "OUT"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """get_neighbors (depth 1) for many nodes: {node_id: the same list, same order}.

        This default is the per-node loop; stores override it with one query.
        """
        return {nid: self.get_neighbors(nid, rel_types, direction=direction) for nid in dict.fromkeys(node_ids)}

    def find_related_memories_batch(self, entity_names: List[str], limit: int = 10) -> Dict[str, List[MemoryNode]]:
        """find_related_memories for many entities: {name: the same list, same order}.

        This default is the per-entity loop; stores override it with one query.
        """
        return {name: self.find_related_memories(name, limit=limit) for name in dict.fromkeys(entity_names)}

    @abstractmethod
    def get_timeline(self, project_id: str, entity: Optional[str] = None, limit: int = 50) -> List[MemoryNode]:
        """Get timeline of memories."""
        pass

    @abstractmethod
    def invalidate_memory(self, id: str, superseded_by: str, reason: str) -> None:
        """Invalidate a memory node."""
        pass

    @abstractmethod
    def iter_memories(self) -> Iterator[MemoryNode]:
        """Yield every memory node, all projects, superseded versions included."""
        pass

    @abstractmethod
    def iter_entities(self) -> Iterator[EntityNode]:
        """Yield every entity node."""
        pass

    @abstractmethod
    def iter_relationships(self) -> Iterator[RelationshipEdge]:
        """Yield every edge between memory/entity nodes, with a stable id."""
        pass

    @abstractmethod
    def restore_memory(self, node: MemoryNode) -> None:
        """Upsert a memory by its id, every field verbatim (no restamping)."""
        pass

    @abstractmethod
    def restore_entity(self, node: EntityNode) -> None:
        """Upsert an entity by its id, every field verbatim."""
        pass

    @abstractmethod
    def restore_relationship(self, edge: RelationshipEdge) -> None:
        """Upsert an edge by its id. Both endpoints must already exist."""
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """Delete every memory, entity and edge (collivind reset)."""
        pass

    @abstractmethod
    def get_version_chain(self, memory_id: str) -> List[MemoryNode]:
        """Walk the supersession chain for a memory, returning all versions oldest-first."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection."""
        pass


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single document text."""
        pass

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query. Some models (e.g. bge) want a query-side
        prefix; providers carry it in config.query_prefix. Documents are
        never prefixed."""
        prefix = getattr(getattr(self, "config", None), "query_prefix", "")
        return self.embed(prefix + text)

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension."""
        pass
