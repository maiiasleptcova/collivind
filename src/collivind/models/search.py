from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .memory import MemoryNode


@dataclass
class SearchQuery:
    query: str
    category: Optional[str] = None
    project_id: str = "default"
    limit: int = 10
    include_graph: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)
    tags: Optional[List[str]] = None
    entity_names: Optional[List[str]] = None
    session_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

@dataclass
class SearchResult:
    memory: MemoryNode
    score: float
    vector_score: float
    graph_score: float = 0.0
    related_entities: List[str] = field(default_factory=list)
    temporal_decay: float = 0.0
