import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


def validate_confidence(value: float) -> float:
    """Coerce and range-check a confidence, or raise ValueError.

    bool is an int subclass, so True would otherwise pass float() as 1.0; NaN
    and inf are rejected by the range test, which both compare False against.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"confidence must be a number between 0 and 1, got {value!r}")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {value!r}")
    return value


class MemoryCategory(str, Enum):
    FACT = "fact"
    DECISION = "decision"
    PATTERN = "pattern"
    ERROR = "error"
    ARCHITECTURE = "architecture"
    PREFERENCE = "preference"
    SNIPPET = "snippet"


class MemorySource(str, Enum):
    HOOK_STOP = "hook_stop"
    HOOK_PRECOMPACT = "hook_precompact"
    PERIODIC = "periodic"
    MANUAL = "manual"


@dataclass
class MemoryNode:
    content: str
    summary: str
    category: MemoryCategory
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = 1.0
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None
    project_id: str = "default"
    session_id: Optional[str] = None
    user_id: str = "local"
    source: MemorySource = MemorySource.MANUAL
    superseded_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    version: int = 1
    previous_version_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "category": self.category.value if isinstance(self.category, Enum) else self.category,
            "confidence": self.confidence,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "source": self.source.value if isinstance(self.source, Enum) else self.source,
            "superseded_by": self.superseded_by,
            "tags": self.tags,
            "version": self.version,
            "previous_version_id": self.previous_version_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class MemoryCreate:
    content: str
    summary: str
    category: MemoryCategory
    project_id: str = "default"
    session_id: Optional[str] = None
    user_id: str = "local"
    source: MemorySource = MemorySource.MANUAL
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate here rather than per-surface.

        Every create path builds a MemoryCreate — CLI `add`, the MCP add and
        batch tools, extraction, `import`, the web POST — so a check in any one
        of them leaves the rest open. `collivind add --confidence nan` reached
        the store and read back as NULL precisely because the guard sat in one
        caller.
        """
        self.confidence = validate_confidence(self.confidence)
