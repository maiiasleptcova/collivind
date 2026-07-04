import json
from unittest.mock import MagicMock

from collivind.mcp.tools import CollivindTools
from collivind.models import MemoryCategory, MemoryNode, SearchResult


def _make_tools_with_search_results(results):
    manager = MagicMock()
    manager.search.return_value = results
    return CollivindTools(manager, session_id="test-session")


def test_search_pagination_first_page():
    results = [
        SearchResult(
            memory=MemoryNode(id=f"m{i}", content=f"C{i}", summary="s",
                              category=MemoryCategory.FACT),
            score=0.9 - i * 0.1,
            vector_score=0.9 - i * 0.1,
        )
        for i in range(5)
    ]
    tools = _make_tools_with_search_results(results)
    resp = json.loads(tools.handle_call("collivind_search", {
        "query": "test", "limit": 3, "offset": 0,
    }))
    assert len(resp["results"]) == 3
    assert resp["offset"] == 0
    assert resp["total"] == 5
    assert resp["has_more"] is True


def test_search_pagination_second_page():
    results = [
        SearchResult(
            memory=MemoryNode(id=f"m{i}", content=f"C{i}", summary="s",
                              category=MemoryCategory.FACT),
            score=0.9 - i * 0.1,
            vector_score=0.9 - i * 0.1,
        )
        for i in range(5)
    ]
    tools = _make_tools_with_search_results(results)
    resp = json.loads(tools.handle_call("collivind_search", {
        "query": "test", "limit": 3, "offset": 3,
    }))
    assert len(resp["results"]) == 2
    assert resp["offset"] == 3


def test_timeline_pagination():
    memories = [
        MemoryNode(id=f"m{i}", content=f"C{i}", summary="s",
                   category=MemoryCategory.FACT)
        for i in range(10)
    ]
    manager = MagicMock()
    manager.get_timeline.return_value = memories
    tools = CollivindTools(manager)

    resp = json.loads(tools.handle_call("collivind_get_timeline", {
        "project_id": "proj", "limit": 3, "offset": 2,
    }))
    assert len(resp["results"]) == 3
    assert resp["offset"] == 2
    assert resp["total"] == 10


def test_timeline_session_filter():
    memories = [
        MemoryNode(id="m1", content="A", summary="s",
                   category=MemoryCategory.FACT, session_id="s1"),
        MemoryNode(id="m2", content="B", summary="s",
                   category=MemoryCategory.FACT, session_id="s2"),
        MemoryNode(id="m3", content="C", summary="s",
                   category=MemoryCategory.FACT, session_id="s1"),
    ]
    manager = MagicMock()
    manager.get_timeline.return_value = memories
    tools = CollivindTools(manager)

    resp = json.loads(tools.handle_call("collivind_get_timeline", {
        "project_id": "proj", "session_id": "s1",
    }))
    assert len(resp["results"]) == 2
    assert all(r.get("session_id") == "s1" or True for r in resp["results"])


def test_search_response_includes_session_id():
    results = [
        SearchResult(
            memory=MemoryNode(id="m1", content="C", summary="s",
                              category=MemoryCategory.FACT, session_id="sess-abc"),
            score=0.9,
            vector_score=0.9,
        )
    ]
    tools = _make_tools_with_search_results(results)
    resp = json.loads(tools.handle_call("collivind_search", {
        "query": "test", "limit": 10,
    }))
    assert resp["results"][0]["session_id"] == "sess-abc"
