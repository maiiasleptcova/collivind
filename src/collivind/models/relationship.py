from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


class RelType(str, Enum):
    # Memory -> Entity
    ABOUT = "ABOUT"
    MENTIONS = "MENTIONS"

    # Memory -> Memory
    SUPERSEDES = "SUPERSEDES"
    CONTRADICTS = "CONTRADICTS"
    RELATES_TO = "RELATES_TO"
    CAUSED_BY = "CAUSED_BY"
    DEPENDS_ON_MEM = "DEPENDS_ON"

    # Entity -> Entity
    IMPLEMENTS = "IMPLEMENTS"
    DEPENDS_ON_ENT = "DEPENDS_ON"
    PART_OF = "PART_OF"
    USES = "USES"
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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RelationshipCreate:
    source_id: str
    target_id: str
    type: RelType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "manual"
