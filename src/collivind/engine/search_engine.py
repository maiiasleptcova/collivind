import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from collivind.config import SearchConfig
from collivind.engine.enrichment import build_enriched_query
from collivind.engine.graph_engine import GraphEngine
from collivind.models import MemoryNode, SearchQuery, SearchResult
from collivind.storage.interfaces import EmbeddingProvider, GraphStore, VectorStore

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_provider: EmbeddingProvider,
        config: SearchConfig,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_provider = embedding_provider
        self.config = config
        self.graph_engine = GraphEngine(graph_store)

    def _compute_temporal_decay(self, memory: MemoryNode, now: datetime) -> float:
        age_days = (now - memory.created_at).total_seconds() / 86400
        decay = 1.0 - self.config.temporal_decay_rate * math.log1p(age_days)
        return max(1.0 - self.config.temporal_decay_max, min(1.0, decay))

    def _passes_filters(self, memory: MemoryNode, query: SearchQuery, now: datetime) -> bool:
        if memory.valid_to and memory.valid_to < now:
            return False
        if query.project_id and memory.project_id != query.project_id:
            return False
        if query.category and memory.category.value != query.category:
            return False
        if query.tags and not set(query.tags).intersection(memory.tags):
            return False
        if query.session_id and memory.session_id != query.session_id:
            return False
        if query.user_id and memory.user_id != query.user_id:
            return False
        if query.date_from and memory.created_at < query.date_from:
            return False
        if query.date_to and memory.created_at > query.date_to:
            return False
        return True

    def search(self, query: SearchQuery) -> List[SearchResult]:
        return self.search_with_status(query)[0]

    def search_with_status(self, query: SearchQuery) -> Tuple[List[SearchResult], bool]:
        """Hybrid search, plus whether it had to degrade to vector-only.

        The graph store is only enrichment for a search: every vector payload
        is the full memory, so when the graph raises the hits are rebuilt from
        the payloads instead of failing the whole read (#39). The vector store
        and embedder stay load-bearing — their errors propagate.
        """
        enriched = build_enriched_query(
            query.query,
            category=query.category,
            tags=query.tags,
            entity_names=query.entity_names,
        )
        vector = self.embedding_provider.embed_query(enriched)

        vector_filters = dict(query.filters)
        if query.project_id:
            vector_filters["project_id"] = query.project_id

        raw_vector_results = self.vector_store.search(
            vector=vector, limit=query.limit * 3, filters=vector_filters, threshold=self.config.similarity_threshold
        )

        now = datetime.now(timezone.utc)
        try:
            candidates = self._graph_candidates(raw_vector_results, query, now)
            degraded = False
        except Exception as e:
            # The first failure is enough: get_memory retries for ~1.5s per
            # call, so pressing on per-hit would turn one dead store into a
            # 45-second search.
            logger.warning("graph store failed during search, serving vector-only results: %s", e)
            candidates = self._vector_only_candidates(raw_vector_results, query, now)
            degraded = True

        results = []
        for res in candidates.values():
            norm_graph_score = min(res.graph_score, 1.0)
            base_score = (res.vector_score * self.config.vector_weight) + (norm_graph_score * self.config.graph_weight)
            res.score = base_score * res.temporal_decay
            results.append(res)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: query.limit], degraded

    def _graph_candidates(self, raw_vector_results, query: SearchQuery, now: datetime) -> Dict[str, SearchResult]:
        candidates: Dict[str, SearchResult] = {}

        for res in raw_vector_results:
            mem_id = res["id"]
            memory = self.graph_store.get_memory(mem_id)
            if not memory:
                continue
            if not self._passes_filters(memory, query, now):
                continue

            decay = self._compute_temporal_decay(memory, now)
            candidates[mem_id] = SearchResult(
                memory=memory,
                score=0.0,
                vector_score=res["score"],
                graph_score=0.0,
                temporal_decay=decay,
            )

        if query.entity_names:
            for ent_name in query.entity_names:
                related = self.graph_store.find_related_memories(ent_name, limit=query.limit)
                for mem in related:
                    if mem.id in candidates or not self._passes_filters(mem, query, now):
                        continue
                    decay = self._compute_temporal_decay(mem, now)
                    candidates[mem.id] = SearchResult(
                        memory=mem,
                        score=0.0,
                        vector_score=0.0,
                        graph_score=0.2,
                        temporal_decay=decay,
                        related_entities=[ent_name],
                    )

        if query.include_graph and candidates:
            expanded = self.graph_engine.get_expanded_memories(list(candidates.keys()))

            for mem_id, data in expanded.items():
                memory = data["memory"]
                if not self._passes_filters(memory, query, now):
                    continue

                shared_entities = data["shared_entities"]

                if mem_id in candidates:
                    candidates[mem_id].graph_score += len(shared_entities) * 0.1
                    candidates[mem_id].related_entities.extend(shared_entities)
                else:
                    decay = self._compute_temporal_decay(memory, now)
                    candidates[mem_id] = SearchResult(
                        memory=memory,
                        score=0.0,
                        vector_score=0.0,
                        graph_score=len(shared_entities) * 0.1,
                        related_entities=shared_entities,
                        temporal_decay=decay,
                    )

        return candidates

    def _vector_only_candidates(self, raw_vector_results, query: SearchQuery, now: datetime) -> Dict[str, SearchResult]:
        """Hits rebuilt from vector payloads alone: no graph score, no entities.

        ponytail: invalidate() updates only the graph, so a payload can still
        claim valid_to=None for a memory invalidated by hand — it will surface
        here until its vector is rewritten. Superseded-by-update memories are
        already deleted from the vector index, so they cannot. Push validity
        into the vector payload on invalidate if that ever matters.
        """
        candidates: Dict[str, SearchResult] = {}
        for res in raw_vector_results:
            try:
                memory = MemoryNode.from_dict({"id": res["id"], **(res.get("payload") or {})})
            except (KeyError, TypeError, ValueError):
                continue
            if memory.superseded_by or not self._passes_filters(memory, query, now):
                continue
            candidates[memory.id] = SearchResult(
                memory=memory,
                score=0.0,
                vector_score=res["score"],
                temporal_decay=self._compute_temporal_decay(memory, now),
            )
        return candidates

    def find_contradictions(self, memory: MemoryNode, threshold: float = 0.75) -> List[SearchResult]:
        vector = self.embedding_provider.embed(memory.content)
        similar = self.vector_store.search(
            vector=vector, limit=5, filters={"project_id": memory.project_id}, threshold=threshold
        )

        candidates = []
        for res in similar:
            existing = self.graph_store.get_memory(res["id"])
            if not existing or existing.id == memory.id:
                continue
            if existing.valid_to is not None:
                continue
            if existing.category == memory.category and existing.content != memory.content:
                candidates.append(
                    SearchResult(memory=existing, score=res["score"], vector_score=res["score"], graph_score=0.0)
                )

        return candidates
