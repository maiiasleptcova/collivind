import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


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
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
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
