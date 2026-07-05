from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


class EntityType(str, Enum):
    PROJECT = "project"
    FILE = "file"
    SERVICE = "service"
    CONCEPT = "concept"
    PERSON = "person"
    LIBRARY = "library"
    TOOL = "tool"


@dataclass
class EntityNode:
    name: str
    type: EntityType
    id: str = ""  # typically lowercased, underscored name
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.id:
            self.id = self.name.lower().replace(" ", "_").replace("-", "_")


@dataclass
class EntityCreate:
    name: str
    type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
