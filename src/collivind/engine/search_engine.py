import logging
import math
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from collivind.config import SearchConfig
from collivind.engine import summaries as summaries_mod
from collivind.engine.conflicts import CONFLICT_THRESHOLD, MAX_CANDIDATES, is_live, same_subject
from collivind.engine.enrichment import build_enriched_query
from collivind.engine.graph_engine import GraphEngine
from collivind.models import MemoryNode, SearchQuery, SearchResult
from collivind.models.memory import MemoryCategory
from collivind.storage.interfaces import EmbeddingProvider, GraphStore, VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60  # reciprocal rank fusion constant (Cormack et al.)
# ponytail: the cross-encoder reads 256 tokens of query + memory; a long
# prompt is cut to its opening so the memory keeps most of the budget.
RERANK_QUERY_CHARS = 1000
# The cross-encoder's own relevance boundary. When no candidate in the pool
# clears it, the model is saying "none of these answer the query" and its
# order among them is noise (indirect prompts: preferences, "what should I
# do about X"), so the first-stage ranking is served instead.
# ponytail: tied to logit-scale models (the MS MARCO cross-encoders); a
# custom rerank_model that outputs 0..1 scores would never trip it.
RERANK_GATE_LOGIT = 0.0

# Summary routing (see _route_summaries). The split mirrors the measured
# "routed" arm: 3 summaries, 15 of their members, 10 other memories.
SUMMARY_CANDIDATES = 20  # dense top-N summaries scored line by line
# ponytail: line scoring is capped at MAX_LINE_PAIRS cross-encoder pairs per
# search; a store whose summaries are longer scores fewer of them. Routing
# measured +150 ms median per prompt recall (127 -> 279 ms, embedded, Apple
# silicon, 60 summaries of 12 lines). Embed summary lines separately if that bites.
MAX_SUMMARY_LINES = 60
MAX_LINE_PAIRS = 300
MAX_ROUTED_MEMBERS = 120
ROUTED_SHARE = 0.6


def _sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))


def _rrf(rankings: List[List[str]], k: int = RRF_K) -> List[str]:
    """Reciprocal rank fusion. Ties keep first-seen order, so dense wins ties."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


def _payload_matches(payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """The vector store's flat MatchValue filter, for candidates that did not come from it."""
    for key, expected in filters.items():
        value = payload.get(key)
        if value != expected and not (isinstance(value, list) and expected in value):
            return False
    return True


class SearchEngine:
    """Dense + graph search, with an optional cross-encoder second stage.

    First stage (0.18.0 ranking, and the fallback): dense top limit*3, graph
    and entity expansion, score = (cosine * vector_weight + min(graph, 1) *
    graph_weight) * temporal_decay.

    Second stage (search.rerank, when the embedding provider has a reranker):
    a pool of rerank_pool candidates fused by reciprocal rank from the dense
    top rerank_pool and the keyword (BM25) top rerank_pool, plus the best
    `limit` graph/entity-only candidates, is scored by the cross-encoder.
    The squashed score sigmoid(logit) takes the cosine's place in the same
    formula, so graph links and temporal decay keep their weight. When no
    pool candidate clears the model's relevance boundary (logit 0), and on
    any reranker problem, the first-stage ranking is returned unchanged.
    """

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

        legacy_n = query.limit * 3
        rerank = self._can_rerank()
        dense = self.vector_store.search(
            vector=vector,
            limit=max(legacy_n, self._pool_size(query)) if rerank else legacy_n,
            filters=vector_filters,
            threshold=self.config.similarity_threshold,
        )
        # the first stage sees exactly what 0.18.0 saw: dense hits are sorted,
        # so the top limit*3 is a prefix of the larger pool fetch
        raw_vector_results = dense[:legacy_n]

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
        if not raw_vector_results and not query.entity_names:
            # Nothing above touched the graph, so it is unknown whether it is
            # up; ask, or an empty answer from a dead store reads as "nothing
            # relevant" — the silent recall #39 was about.
            degraded = not self._graph_healthy()

        results = []
        for res in candidates.values():
            norm_graph_score = min(res.graph_score, 1.0)
            base_score = (res.vector_score * self.config.vector_weight) + (norm_graph_score * self.config.graph_weight)
            res.score = base_score * res.temporal_decay
            results.append(res)

        results.sort(key=lambda x: x.score, reverse=True)
        if rerank and not degraded:
            reranked = self._rerank(query, candidates, results, dense, now)
            if reranked is not None:
                routed = self._route_summaries(query, vector, vector_filters, reranked, now)
                return (routed if routed is not None else reranked[: query.limit]), degraded
        return results[: query.limit], degraded

    def _can_rerank(self) -> bool:
        try:
            return bool(self.config.rerank) and getattr(self.embedding_provider, "can_rerank", False) is True
        except Exception:
            return False

    def _pool_size(self, query: SearchQuery) -> int:
        return max(self.config.rerank_pool, query.limit)

    def _rerank(
        self,
        query: SearchQuery,
        candidates: Dict[str, SearchResult],
        first_stage: List[SearchResult],
        dense: List[Dict[str, Any]],
        now: datetime,
    ) -> Optional[List[SearchResult]]:
        """Second-stage ranking, or None to keep the first stage unchanged. Never raises.

        Ordering and membership are separate decisions. The gate picks the
        order (reranked, or first stage when no candidate is confidently
        relevant); query.min_rerank_score then drops every result whose
        cross-encoder score is below it, in either order, so a recall with
        nothing relevant returns fewer results, possibly none.
        """
        try:
            pool = self._rerank_pool(query, candidates, first_stage, dense, now)
            if not pool:
                return None
            texts = [r.memory.content for r in pool]
            logits = self.embedding_provider.rerank(query.query[:RERANK_QUERY_CHARS], texts)
            if not isinstance(logits, list) or len(logits) != len(pool):
                return None
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in logits):
                return None  # a NaN would squash to 1.0 and rank first
            cutoff = query.min_rerank_score or 0.0
            if max(logits) < RERANK_GATE_LOGIT:
                if not cutoff:
                    return None
                squashed = {r.memory.id: _sigmoid(float(x)) for r, x in zip(pool, logits)}
                # the pool holds the whole first-stage top `limit`, so each has a score
                return [
                    replace(r, rerank_score=squashed[r.memory.id])
                    for r in first_stage
                    if squashed.get(r.memory.id, 0.0) >= cutoff
                ]
            out = []
            for res, logit in zip(pool, logits):
                rerank_score = _sigmoid(float(logit))
                base = rerank_score * self.config.vector_weight + min(res.graph_score, 1.0) * self.config.graph_weight
                out.append(replace(res, score=base * res.temporal_decay, rerank_score=rerank_score))
            out.sort(key=lambda x: x.score, reverse=True)
            return [r for r in out if (r.rerank_score or 0.0) >= cutoff]
        except Exception as e:
            logger.warning("rerank failed, serving first-stage ranking: %s", e)
            return None

    def _route_summaries(
        self,
        query: SearchQuery,
        vector: List[float],
        vector_filters: Dict[str, Any],
        reranked: List[SearchResult],
        now: datetime,
    ) -> Optional[List[SearchResult]]:
        """Summaries first, then the details inside them ("routed", BENCHMARKS.md).

        The best summaries for the query are ranked against each other (they
        would lose a pooled ranking to short memories that echo the query),
        then their live member memories are reranked against the query. A
        summary routes only when its best line clears the reranker's own
        relevance boundary, so an off-topic summary never leads a result.
        The result is the routed summaries, then up to ~60% of the remaining
        slots from their members and the rest from the ordinary ranking,
        those two merged by score. None, so the caller keeps the ordinary
        ranking unchanged, when no summary qualifies: a store without
        summaries searches exactly as before. Never raises.
        """
        routes = self.config.summary_routes
        if routes <= 0 or query.category not in (None, MemoryCategory.SUMMARY.value) or query.limit < 2:
            return None
        try:
            summaries = self._summary_candidates(query, vector, vector_filters, reranked, now)
            if not summaries:
                return None
            cutoff = query.min_rerank_score or 0.0
            # ponytail: a quarter of the slots, at most `routes`; 3 of 28 is what was measured
            gate = max(cutoff, _sigmoid(RERANK_GATE_LOGIT))
            top = [r for r in self._rescore(query, summaries, by_line=True) if (r.rerank_score or 0.0) >= gate]
            top = top[: min(routes, max(1, query.limit // 4))]
            if not top:
                return None
            members = summaries_mod.members_of(self, [r.memory.id for r in top], cap=MAX_ROUTED_MEMBERS)
            seen = {r.memory.id for r in top}
            pool = []
            for r in top:
                for m in members.get(r.memory.id, []):
                    if m.id not in seen and self._passes_filters(m, query, now):
                        seen.add(m.id)
                        pool.append(
                            SearchResult(
                                memory=m,
                                score=0.0,
                                vector_score=0.0,
                                temporal_decay=self._compute_temporal_decay(m, now),
                            )
                        )
            # ponytail: members are capped so one huge group cannot make the second rerank slow
            details = [r for r in self._rescore(query, pool[:MAX_ROUTED_MEMBERS]) if (r.rerank_score or 0.0) >= cutoff]
            slots = query.limit - len(top)
            routed = details[: math.ceil(slots * ROUTED_SHARE)]
            taken = {r.memory.id for r in top + routed}
            others = [r for r in reranked if r.memory.id not in taken and not summaries_mod.is_summary(r.memory)]
            rest = routed + others[: slots - len(routed)]
            rest += [r for r in details if r.memory.id not in {x.memory.id for x in rest}][: slots - len(rest)]
            # membership follows the measured split; order follows the scores, so a
            # weak member never sits above a strong ordinary match
            return top + sorted(rest, key=lambda r: r.score, reverse=True)
        except Exception as e:
            logger.warning("summary routing failed, serving the ordinary ranking: %s", e)
            return None

    def _summary_candidates(
        self,
        query: SearchQuery,
        vector: List[float],
        vector_filters: Dict[str, Any],
        reranked: List[SearchResult],
        now: datetime,
    ) -> List[SearchResult]:
        """Live summaries near the query: a dense fetch restricted to summaries, plus any already ranked."""
        hits = self.vector_store.search(
            vector=vector,
            limit=SUMMARY_CANDIDATES,
            filters={**vector_filters, "category": MemoryCategory.SUMMARY.value},
            threshold=self.config.similarity_threshold,
        )
        found: Dict[str, SearchResult] = {r.memory.id: r for r in reranked if summaries_mod.is_summary(r.memory)}
        for hit in hits:
            if hit["id"] in found:
                continue
            memory = self.graph_store.get_memory(hit["id"])
            if not summaries_mod.is_live(memory) or memory is None or not summaries_mod.is_summary(memory):
                continue
            if not self._passes_filters(memory, query, now):
                continue
            found[memory.id] = SearchResult(
                memory=memory,
                score=0.0,
                vector_score=hit["score"],
                temporal_decay=self._compute_temporal_decay(memory, now),
            )
        return sorted(found.values(), key=lambda r: r.vector_score, reverse=True)

    def _rescore(self, query: SearchQuery, pool: List[SearchResult], by_line: bool = False) -> List[SearchResult]:
        """Cross-encoder scores for `pool`, best first; raises when the reranker gives none.

        by_line scores each line of a memory on its own and keeps the best:
        the cross-encoder reads 256 tokens, so a summary of a few hundred
        words would otherwise be judged on its first facts only (on LoCoMo
        the right summary reached the top 3 for 207 of 300 questions scored
        whole, 277 scored by its best line).
        """
        if not pool:
            return []
        owners: List[int] = []
        texts: List[str] = []
        kept: List[SearchResult] = []
        for r in pool:  # best dense match first, so the pair budget drops the far end
            parts = [ln for ln in r.memory.content.splitlines() if ln.strip()][:MAX_SUMMARY_LINES] if by_line else []
            parts = parts or [r.memory.content]
            if by_line and kept and len(texts) + len(parts) > MAX_LINE_PAIRS:
                break
            for text in parts:
                owners.append(len(kept))
                texts.append(text)
            kept.append(r)
        pool = kept
        raw = self.embedding_provider.rerank(query.query[:RERANK_QUERY_CHARS], texts)
        if not isinstance(raw, list) or len(raw) != len(texts):
            raise RuntimeError("reranker returned no scores")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in raw):
            raise RuntimeError("reranker returned a non-finite score")
        logits = [float("-inf")] * len(pool)
        for i, x in zip(owners, raw):
            logits[i] = max(logits[i], float(x))
        out = []
        for res, logit in zip(pool, logits):
            rs = _sigmoid(float(logit))
            out.append(replace(res, score=rs * self.config.vector_weight * res.temporal_decay, rerank_score=rs))
        out.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        return out

    def _rerank_pool(
        self,
        query: SearchQuery,
        candidates: Dict[str, SearchResult],
        first_stage: List[SearchResult],
        dense: List[Dict[str, Any]],
        now: datetime,
    ) -> List[SearchResult]:
        """RRF(dense, keyword) top pool_size, plus the best `limit` graph/entity-only
        candidates, plus the whole first-stage top `limit` (so the gate's
        fallback order can be filtered by score too)."""
        size = self._pool_size(query)
        dense_scores = {r["id"]: r["score"] for r in dense}
        lexical = self._lexical_candidates(query, size, now)
        pool: Dict[str, SearchResult] = {}
        for mem_id in _rrf([list(dense_scores), list(lexical)])[:size]:
            if mem_id in candidates:
                pool[mem_id] = candidates[mem_id]
                continue
            memory = lexical.get(mem_id) or self.graph_store.get_memory(mem_id)
            if not is_live(memory) or memory is None or not self._passes_filters(memory, query, now):
                continue
            pool[mem_id] = SearchResult(
                memory=memory,
                score=0.0,
                vector_score=dense_scores.get(mem_id, 0.0),
                temporal_decay=self._compute_temporal_decay(memory, now),
            )
        graph_only = sorted(
            (r for r in candidates.values() if r.vector_score == 0.0 and r.memory.id not in pool),
            key=lambda r: r.score,
            reverse=True,
        )
        for res in graph_only[: query.limit] + first_stage[: query.limit]:
            pool.setdefault(res.memory.id, res)
        return list(pool.values())

    def _lexical_candidates(self, query: SearchQuery, limit: int, now: datetime) -> Dict[str, MemoryNode]:
        """Keyword hits that pass every search filter, best first.

        ponytail: project and liveness are filtered in the store before its
        LIMIT; category/tags/session/user/dates after it, so a narrow filter
        can leave fewer than `limit`. Push them into the store if that bites.
        """
        if not self.config.lexical:
            return {}
        try:
            hits = self.graph_store.lexical_search(query.query, limit, project_id=query.project_id or None)
        except Exception as e:  # the store contract says it never raises; a third-party store might
            logger.warning("keyword search failed, using dense candidates only: %s", e)
            return {}
        if not isinstance(hits, list):
            return {}
        return {
            memory.id: memory
            for memory, _ in hits
            if is_live(memory)
            and self._passes_filters(memory, query, now)
            and _payload_matches(memory.to_dict(), query.filters)
        }

    def _graph_healthy(self) -> bool:
        try:
            return self.graph_store.health_check().get("status") == "ok"
        except Exception:
            return False

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

        Invalidated, superseded and forgotten memories have no vector (the
        manager deletes it). ponytail: a memory invalidated by hand before
        0.11.1 kept its vector and a live-looking payload, so it can still
        surface here while the graph is down; `collivind migrate` drops such
        vectors, add a one-off sweep if that double fault ever matters.
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

    def find_contradictions(self, memory: MemoryNode, threshold: float = CONFLICT_THRESHOLD) -> List[SearchResult]:
        """Live memories a person or agent should check against `memory` (see engine/conflicts.py)."""
        return self.conflict_candidates(memory, self.embedding_provider.embed(memory.content), threshold)

    def conflict_candidates(
        self,
        memory: MemoryNode,
        vector: List[float],
        threshold: float = CONFLICT_THRESHOLD,
        exclude: Tuple[str, ...] = (),
    ) -> List[SearchResult]:
        """Same-project, same-category live memories that are close to `memory` and about the same subject.

        Pass the vector the memory was stored with, so scores are on the
        same scale as the threshold. Similarity alone cannot tell an update
        from a conflict from a neighbour, so these are candidates for the
        caller to judge, never edges.
        """
        similar = self.vector_store.search(
            vector=vector, limit=MAX_CANDIDATES + 3, filters={"project_id": memory.project_id}, threshold=threshold
        )
        candidates: List[SearchResult] = []
        for res in similar:
            if res["id"] == memory.id or res["id"] in exclude:
                continue
            existing = self.graph_store.get_memory(res["id"])
            if not is_live(existing) or existing is None:
                continue
            if existing.category != memory.category or existing.content == memory.content:
                continue
            if not same_subject(existing.content, memory.content):
                continue
            candidates.append(
                SearchResult(memory=existing, score=res["score"], vector_score=res["score"], graph_score=0.0)
            )
        return candidates[:MAX_CANDIDATES]
