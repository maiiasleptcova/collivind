import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

@dataclass
class SessionNode:
    project_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    summary: str = ""
    memory_count: int = 0
