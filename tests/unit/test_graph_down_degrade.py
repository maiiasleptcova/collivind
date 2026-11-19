"""Graph store down, vector store up: search degrades, writes fail loudly (#39).

For 46 hours Neo4j crash-looped while Qdrant held every embedding *and* the
full memory payload, yet every search failed because each hit was re-read
from the graph. These pin the vector-only fallback and its limits.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from collivind.config import CollivindConfig, SearchConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.engine.search_engine import SearchEngine
from collivind.exceptions import CollivindError, GraphStoreUnavailableError
from collivind.mcp.tools import CollivindTools
from collivind.models import MemoryCategory, MemoryCreate, MemoryNode, SearchQuery


def _payload(mem_id, content, **overrides):
    node = MemoryNode(content=content, summary=content[:20], category=MemoryCategory.DECISION, id=mem_id)
    return {**node.to_dict(), **overrides}


def _hits(*payloads, score=0.9):
    return [{"id": p["id"], "score": score, "payload": p} for p in payloads]


def _dead_graph():
    graph = MagicMock()
    for method in (
        "get_memory",
        "find_related_memories",
        "find_related_memories_batch",
        "get_neighbors",
        "get_neighbors_batch",
        "create_memory",
        "invalidate_memory",
    ):
        getattr(graph, method).side_effect = CollivindError("Couldn't connect to localhost:7687")
    graph.health_check.return_value = {"status": "error", "message": "Connection refused"}
    return graph


def _live_graph():
    graph = MagicMock()
    graph.health_check.return_value = {"status": "ok"}
    return graph


def _engine(vector_store, graph_store):
    embed = MagicMock()
    embed.embed_query.return_value = [0.1, 0.2]
    return SearchEngine(vector_store, graph_store, embed, SearchConfig())


def _manager(vector_store, graph_store):
    embed = MagicMock()
    embed.embed_query.return_value = [0.1, 0.2]
    embed.embed.return_value = [0.1, 0.2]
    return MemoryManager(vector_store, graph_store, embed, CollivindConfig())


class TestVectorOnlySearch:
    def test_serves_hits_from_the_payload_when_the_graph_raises(self):
        vector = MagicMock()
        vector.search.return_value = _hits(_payload("m1", "use uv not pip"), _payload("m2", "ruff for lint"))

        results, degraded = _engine(vector, _dead_graph()).search_with_status(
            SearchQuery(query="tooling", entity_names=["uv"])
        )

        assert degraded is True
        assert {r.memory.id for r in results} == {"m1", "m2"}
        assert results[0].memory.category is MemoryCategory.DECISION
        assert isinstance(results[0].memory.created_at, datetime)

    def test_stops_calling_the_graph_after_the_first_failure(self):
        """get_memory retries ~1.5s per call; 30 candidates must not mean 45s."""
        vector = MagicMock()
        vector.search.return_value = _hits(*[_payload(f"m{i}", f"c{i}") for i in range(10)])
        graph = _dead_graph()

        _engine(vector, graph).search_with_status(SearchQuery(query="x", entity_names=["e"]))

        assert graph.get_memory.call_count == 1
        graph.find_related_memories.assert_not_called()
        graph.find_related_memories_batch.assert_not_called()

    def test_drops_invalidated_superseded_and_other_project_payloads(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        vector = MagicMock()
        vector.search.return_value = _hits(
            _payload("live", "current"),
            _payload("expired", "old", valid_to=past),
            _payload("replaced", "older", superseded_by="live"),
            _payload("elsewhere", "other", project_id="other-project"),
        )

        results, _ = _engine(vector, _dead_graph()).search_with_status(SearchQuery(query="x", project_id="default"))

        assert [r.memory.id for r in results] == ["live"]
        assert vector.search.call_args.kwargs["filters"]["project_id"] == "default"

    def test_skips_payloads_too_broken_to_rebuild(self):
        vector = MagicMock()
        vector.search.return_value = [{"id": "x", "score": 0.9, "payload": {"id": "x"}}] + _hits(_payload("ok", "fine"))

        results, degraded = _engine(vector, _dead_graph()).search_with_status(SearchQuery(query="x"))

        assert degraded and [r.memory.id for r in results] == ["ok"]

    def test_healthy_graph_is_not_degraded(self):
        vector = MagicMock()
        vector.search.return_value = _hits(_payload("m1", "c"))
        graph = MagicMock()
        graph.get_memory.return_value = MemoryNode(content="c", summary="c", category=MemoryCategory.FACT, id="m1")
        engine = _engine(vector, graph)
        engine.graph_engine = MagicMock()
        engine.graph_engine.get_expanded_memories.return_value = {}

        results, degraded = engine.search_with_status(SearchQuery(query="x"))

        assert degraded is False and len(results) == 1

    def test_a_vector_store_failure_still_raises(self):
        vector = MagicMock()
        vector.search.side_effect = CollivindError("Qdrant search failed")
        with pytest.raises(CollivindError, match="Qdrant"):
            _engine(vector, _dead_graph()).search_with_status(SearchQuery(query="x"))


class TestDegradedOutput:
    def test_get_context_says_graph_enrichment_is_missing(self):
        vector = MagicMock()
        vector.search.return_value = _hits(_payload("m1", "use uv not pip"))

        context = _manager(vector, _dead_graph()).get_context("tooling")

        assert "use uv not pip" in context
        assert "degraded" in context.lower() and "collivind status" in context

    def test_zero_hits_from_a_dead_graph_still_say_degraded(self):
        """No hit means the graph was never asked; an empty answer must not
        read as "nothing relevant" when the store is in fact down."""
        vector = MagicMock()
        vector.search.return_value = []

        context = _manager(vector, _dead_graph()).get_context("x")

        assert "No relevant context" in context and "degraded" in context.lower()

    def test_zero_hits_from_a_live_graph_are_not_degraded(self):
        vector = MagicMock()
        vector.search.return_value = []
        assert "degraded" not in _manager(vector, _live_graph()).get_context("x").lower()

    def test_mcp_search_flags_degraded_with_empty_results(self):
        """The flag must not ride on results alone: an entity lookup that hits
        the dead graph with zero vector hits still has to say so."""
        vector = MagicMock()
        vector.search.return_value = []
        tools = CollivindTools(_manager(vector, _dead_graph()))

        out = json.loads(tools.handle_call("collivind_search", {"query": "x", "entity_names": ["uv"]}))

        assert out["degraded"] is True and out["results"] == []

    def test_mcp_search_marks_the_response(self):
        vector = MagicMock()
        vector.search.return_value = _hits(_payload("m1", "use uv not pip"))
        tools = CollivindTools(_manager(vector, _dead_graph()))

        out = json.loads(tools.handle_call("collivind_search", {"query": "tooling"}))

        assert out["degraded"] is True
        assert "collivind status" in out["warning"]
        assert out["results"][0]["id"] == "m1"

    def test_mcp_search_healthy_reports_not_degraded(self):
        vector = MagicMock()
        vector.search.return_value = []
        out = json.loads(
            CollivindTools(_manager(vector, _live_graph())).handle_call("collivind_search", {"query": "x"})
        )
        assert out["degraded"] is False and "warning" not in out


class TestWritesFailLoudly:
    def test_add_memory_names_the_graph_store_and_status(self):
        vector = MagicMock()
        vector.search.return_value = []
        manager = _manager(vector, _dead_graph())

        with pytest.raises(GraphStoreUnavailableError) as exc:
            manager.add_memory(MemoryCreate(content="c", summary="s", category=MemoryCategory.FACT))

        msg = str(exc.value)
        assert "graph store" in msg.lower() and "collivind status" in msg and "not saved" in msg.lower()
        vector.upsert.assert_not_called()  # no half-write into the vector index

    def test_mcp_add_memory_returns_the_error(self):
        vector = MagicMock()
        vector.search.return_value = []
        tools = CollivindTools(_manager(vector, _dead_graph()))

        out = json.loads(
            tools.handle_call(
                "collivind_add_memory", {"content": "c", "summary": "s", "category": "fact", "project_id": "default"}
            )
        )

        assert "graph store" in out["error"].lower() and "collivind status" in out["error"]

    @pytest.mark.parametrize(
        "call, outcome",
        [
            (lambda m: m.invalidate("m1", "m2", "r"), "nothing was invalidated"),
            (lambda m: m.forget("m1"), "nothing was deleted"),
            (lambda m: m.update_memory("m1", tags=["a"]), "nothing was updated"),
        ],
    )
    def test_other_writes_name_the_graph_store_too(self, call, outcome):
        with pytest.raises(GraphStoreUnavailableError, match=outcome) as exc:
            call(_manager(MagicMock(), _dead_graph()))
        assert "collivind status" in str(exc.value)

    def test_mcp_forget_goes_through_the_guard(self):
        out = json.loads(
            CollivindTools(_manager(MagicMock(), _dead_graph())).handle_call("collivind_forget", {"memory_id": "m1"})
        )
        assert "nothing was deleted" in out["error"]

    def test_raw_driver_errors_get_the_status_hint(self):
        """Most Neo4j reads raise the driver's own ServiceUnavailable, not a
        CollivindError, and used to come back with no pointer at all."""

        class ServiceUnavailable(Exception):
            pass

        graph = _dead_graph()
        graph.get_timeline.side_effect = ServiceUnavailable("Couldn't connect to localhost:7687")
        out = json.loads(CollivindTools(_manager(MagicMock(), graph)).handle_call("collivind_get_timeline", {}))
        assert "collivind status" in out["hint"]

    def test_argument_errors_do_not_blame_the_backend(self):
        out = json.loads(CollivindTools(_manager(MagicMock(), _live_graph())).handle_call("collivind_forget", {}))
        assert "hint" not in out


class TestFactoryToleratesADeadGraph:
    class _Alive:
        def health_check(self):
            return {"status": "ok"}

    class _Dead:
        def health_check(self):
            return {"status": "error", "message": "connection refused"}

    def _patch(self, monkeypatch, vector, graph):
        from collivind.storage import factory

        monkeypatch.setattr(factory, "create_vector_store", lambda c: vector)
        monkeypatch.setattr(factory, "create_graph_store", lambda c: graph)
        monkeypatch.setattr(factory, "create_embedding_provider", lambda c: self._Alive())
        return factory

    def test_opt_in_returns_backends_with_only_the_graph_down(self, monkeypatch):
        factory = self._patch(monkeypatch, self._Alive(), self._Dead())
        assert len(factory.create_all_backends(CollivindConfig(), allow_graph_down=True)) == 3

    def test_default_stays_strict(self, monkeypatch):
        factory = self._patch(monkeypatch, self._Alive(), self._Dead())
        with pytest.raises(factory.ModeUnavailableError):
            factory.create_all_backends(CollivindConfig())

    def test_opt_in_still_rejects_a_dead_vector_store(self, monkeypatch):
        factory = self._patch(monkeypatch, self._Dead(), self._Dead())
        with pytest.raises(factory.ModeUnavailableError):
            factory.create_all_backends(CollivindConfig(), allow_graph_down=True)

    def test_mcp_server_and_cli_manager_opt_in(self, monkeypatch):
        """The degrade is unreachable if the entry points refuse to build."""
        from collivind.cli.commands import memory as memory_cmd
        from collivind.mcp import server as mcp_server

        seen = []

        def fake(config, allow_graph_down=False):
            seen.append(allow_graph_down)
            return MagicMock(), MagicMock(), MagicMock()

        for module in (memory_cmd, mcp_server):
            monkeypatch.setattr(module, "create_all_backends", fake)
            monkeypatch.setattr(module, "load_config", CollivindConfig)  # never the real ~/.collivind
        memory_cmd._manager()
        mcp_server.MCPServer()
        assert seen == [True, True]
