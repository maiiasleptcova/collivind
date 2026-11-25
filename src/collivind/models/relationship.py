import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class RelType(str, Enum):
    # Memory -> Entity
    ABOUT = "ABOUT"
    MENTIONS = "MENTIONS"

    # Memory -> Memory
    SUPERSEDES = "SUPERSEDES"
    CONTRADICTS = "CONTRADICTS"
    RELATES_TO = "RELATES_TO"
    CAUSED_BY = "CAUSED_BY"
    RESOLVES = "RESOLVES"
    DEPENDS_ON_MEM = "DEPENDS_ON"
    SUMMARIZES = "SUMMARIZES"  # summary memory -> each member (detail) memory

    # Entity -> Entity
    IMPLEMENTS = "IMPLEMENTS"
    DEPENDS_ON_ENT = "DEPENDS_ON"
    PART_OF = "PART_OF"
    USES = "USES"
    INTERACTS_WITH = "INTERACTS_WITH"
    RELATED_TO = "RELATED_TO"

    # Session
    PRODUCED = "PRODUCED"
    WORKED_ON = "WORKED_ON"


@dataclass
class RelationshipEdge:
    id: str = ""
    source_id: str = ""
    target_id: str = ""
    type: RelType = RelType.RELATES_TO
    confidence: float = 1.0
    source: str = "manual"
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )  # may be absent in old rows


@dataclass
class RelationshipCreate:
    source_id: str
    target_id: str
    type: RelType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "manual"


_EDGE_NAMESPACE = uuid.UUID("6f1c2a52-6a3e-4d1b-9a57-7c0f3b1e2d47")


def stable_edge_id(source_id: str, rel_type: str, target_id: str) -> str:
    """Deterministic id for an edge stored without one (Neo4j MERGEs on the triple)."""
    return str(uuid.uuid5(_EDGE_NAMESPACE, f"{source_id}|{rel_type}|{target_id}"))
