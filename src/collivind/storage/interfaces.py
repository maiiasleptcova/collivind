from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
        self, vector: List[float], limit: int = 10,
        filters: Optional[Dict[str, Any]] = None, threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a vector by id."""
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
        self, node_id: str, rel_types: List[str],
        direction: str = "OUT", depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes in the graph."""
        pass

    @abstractmethod
    def find_related_memories(self, entity_name: str, limit: int = 10) -> List[MemoryNode]:
        """Find memories related to an entity."""
        pass

    @abstractmethod
    def get_timeline(self, project_id: str, entity: Optional[str] = None, limit: int = 50) -> List[MemoryNode]:
        """Get timeline of memories."""
        pass

    @abstractmethod
    def invalidate_memory(self, id: str, superseded_by: str, reason: str) -> None:
        """Invalidate a memory node."""
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
        """Embed a single text."""
        pass

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
