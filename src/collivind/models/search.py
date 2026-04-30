from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .memory import MemoryNode

@dataclass
class SearchQuery:
    query: str
    category: Optional[str] = None
    project_id: str = "default"
    limit: int = 10
    include_graph: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    memory: MemoryNode
    score: float
    vector_score: float
    graph_score: float = 0.0
    related_entities: List[str] = field(default_factory=list)
