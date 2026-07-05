from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from collivind.config import SearchConfig
from collivind.engine.search_engine import SearchEngine
from collivind.mcp.server import MCPServer
from collivind.models import MemoryCategory, MemoryNode, SearchQuery


def test_server_generates_session_id():
    with patch("collivind.mcp.server.create_all_backends") as mock_b, patch("collivind.mcp.server.load_config"):
        mock_b.return_value = (MagicMock(), MagicMock(), MagicMock())
        server = MCPServer()
        assert server.session_id
        assert len(server.session_id) == 36  # UUID format


def test_tools_receives_session_id():
    with patch("collivind.mcp.server.create_all_backends") as mock_b, patch("collivind.mcp.server.load_config"):
        mock_b.return_value = (MagicMock(), MagicMock(), MagicMock())
        server = MCPServer()
        assert server.tools.session_id == server.session_id


def test_session_filter_in_search():
    now = datetime.now(timezone.utc)
    m1 = MemoryNode(
        id="m1", content="A", summary="s", category=MemoryCategory.FACT, session_id="sess-1", created_at=now
    )
    m2 = MemoryNode(
        id="m2", content="B", summary="s", category=MemoryCategory.FACT, session_id="sess-2", created_at=now
    )

    vs = MagicMock()
    gs = MagicMock()
    ep = MagicMock()
    ep.embed.return_value = [0.1, 0.2]
    vs.search.return_value = [
        {"id": "m1", "score": 0.8},
        {"id": "m2", "score": 0.8},
    ]
    gs.get_memory.side_effect = lambda mid: m1 if mid == "m1" else m2

    config = SearchConfig()
    engine = SearchEngine(vs, gs, ep, config)
    engine.graph_engine = MagicMock()
    engine.graph_engine.get_expanded_memories.return_value = {}

    results = engine.search(SearchQuery(query="test", session_id="sess-1", limit=10))
    assert len(results) == 1
    assert results[0].memory.id == "m1"


def test_no_session_filter_returns_all():
    now = datetime.now(timezone.utc)
    m1 = MemoryNode(
        id="m1", content="A", summary="s", category=MemoryCategory.FACT, session_id="sess-1", created_at=now
    )
    m2 = MemoryNode(
        id="m2", content="B", summary="s", category=MemoryCategory.FACT, session_id="sess-2", created_at=now
    )

    vs = MagicMock()
    gs = MagicMock()
    ep = MagicMock()
    ep.embed.return_value = [0.1, 0.2]
    vs.search.return_value = [
        {"id": "m1", "score": 0.8},
        {"id": "m2", "score": 0.8},
    ]
    gs.get_memory.side_effect = lambda mid: m1 if mid == "m1" else m2

    config = SearchConfig()
    engine = SearchEngine(vs, gs, ep, config)
    engine.graph_engine = MagicMock()
    engine.graph_engine.get_expanded_memories.return_value = {}

    results = engine.search(SearchQuery(query="test", limit=10))
    assert len(results) == 2
