import json
from unittest.mock import MagicMock

from collivind.mcp.tools import CollivindTools


def test_tools_list():
    manager = MagicMock()
    tools = CollivindTools(manager)
    tool_list = tools.get_tool_list()

    assert len(tool_list) == 13
    names = [t["name"] for t in tool_list]
    assert "collivind_status" in names
    assert "collivind_add_memory" in names
    assert "collivind_batch_add" in names
    assert "collivind_find_contradictions" in names


def test_add_memory_tool():
    manager = MagicMock()
    mock_mem = MagicMock()
    mock_mem.id = "test-mem-1"
    manager.add_memory.return_value = mock_mem

    tools = CollivindTools(manager)
    args = {
        "content": "Test memory",
        "summary": "Test",
        "category": "fact",
        "entities": [{"name": "TestEntity", "type": "concept"}],
        "relationships": [],
    }

    response = tools.handle_call("collivind_add_memory", args)
    res_dict = json.loads(response)

    assert res_dict["status"] == "success"
    assert res_dict["memory_id"] == "test-mem-1"
    manager.add_memory.assert_called_once()


def test_batch_add_tool():
    manager = MagicMock()
    manager.batch_add_memories.return_value = ["mem-1", "mem-2"]

    tools = CollivindTools(manager, session_id="session-123")
    args = {
        "memories": [
            {"content": "First", "summary": "Sum 1", "category": "fact"},
            {"content": "Second", "summary": "Sum 2", "category": "fact"},
        ]
    }

    response = tools.handle_call("collivind_batch_add", args)
    res_dict = json.loads(response)

    assert res_dict["status"] == "success"
    assert "mem-1" in res_dict["memory_ids"]
    manager.batch_add_memories.assert_called_once()
    saved_memories = manager.batch_add_memories.call_args.args[0]
    assert saved_memories[0]["session_id"] == "session-123"
    assert saved_memories[1]["session_id"] == "session-123"


def test_status_uses_configured_backends():
    manager = MagicMock()
    manager.config.mode = "embedded"
    manager.vector_store.health_check.return_value = {"status": "ok", "message": "vector"}
    manager.graph_store.health_check.return_value = {"status": "ok", "message": "graph"}
    manager.embedding_provider.health_check.return_value = {"status": "ok", "message": "embedding"}

    tools = CollivindTools(manager)
    response = tools.handle_call("collivind_status", {})
    data = json.loads(response)

    assert data["mode"] == "embedded"
    assert data["vector_store"]["message"] == "vector"
    assert data["graph_store"]["message"] == "graph"
    assert data["embedding_provider"]["message"] == "embedding"
