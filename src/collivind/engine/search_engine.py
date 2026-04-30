from typing import List, Dict, Any
from datetime import datetime, timezone

from collivind.config import SearchConfig
from collivind.storage.interfaces import VectorStore, EmbeddingProvider, GraphStore
from collivind.engine.graph_engine import GraphEngine
from collivind.models import SearchQuery, SearchResult, MemoryNode

class SearchEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_provider: EmbeddingProvider,
        config: SearchConfig
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_provider = embedding_provider
        self.config = config
        self.graph_engine = GraphEngine(graph_store)

    def search(self, query: SearchQuery) -> List[SearchResult]:
        # 1. Vector Search
        vector = self.embedding_provider.embed(query.query)
        
        # We fetch more candidates than limit to allow re-ranking and filtering
        raw_vector_results = self.vector_store.search(
            vector=vector,
            limit=query.limit * 2,
            filters=query.filters,
            threshold=self.config.similarity_threshold
        )
        
        candidates: Dict[str, SearchResult] = {}
        
        now = datetime.now(timezone.utc)
        
        for res in raw_vector_results:
            mem_id = res["id"]
            memory = self.graph_store.get_memory(mem_id)
            if not memory:
                continue
                
            # Temporal filtering: skip if superseded or valid_to is in the past
            if memory.valid_to and memory.valid_to < now:
                continue
                
            # Filter by category if requested
            if query.category and memory.category != query.category:
                continue
                
            candidates[mem_id] = SearchResult(
                memory=memory,
                score=0.0, # Computed later
                vector_score=res["score"],
                graph_score=0.0
            )
            
        # 2. Graph Expansion
        if query.include_graph and candidates:
            expanded = self.graph_engine.get_expanded_memories(list(candidates.keys()))
            
            for mem_id, data in expanded.items():
                memory = data["memory"]
                
                # Temporal filtering
                if memory.valid_to and memory.valid_to < now:
                    continue
                if query.category and memory.category != query.category:
                    continue
                    
                shared_entities = data["shared_entities"]
                
                if mem_id in candidates:
                    candidates[mem_id].graph_score += len(shared_entities) * 0.1
                    candidates[mem_id].related_entities.extend(shared_entities)
                else:
                    candidates[mem_id] = SearchResult(
                        memory=memory,
                        score=0.0,
                        vector_score=0.0, # Baseline penalty, or fetch vector score
                        graph_score=len(shared_entities) * 0.1,
                        related_entities=shared_entities
                    )

        # 3. Re-ranking
        results = []
        for res in candidates.values():
            # Normalize graph score slightly (cap at 1.0)
            norm_graph_score = min(res.graph_score, 1.0)
            
            res.score = (
                (res.vector_score * self.config.vector_weight) +
                (norm_graph_score * self.config.graph_weight)
            )
            results.append(res)
            
        # Sort by final score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:query.limit]
