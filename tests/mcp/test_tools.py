import json
import pytest
from unittest.mock import MagicMock
from collivind.mcp.tools import CollivindTools

def test_tools_list():
    manager = MagicMock()
    tools = CollivindTools(manager)
    tool_list = tools.get_tool_list()
    
    assert len(tool_list) == 10
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
        "relationships": []
    }
    
    response = tools.handle_call("collivind_add_memory", args)
    res_dict = json.loads(response)
    
    assert res_dict["status"] == "success"
    assert res_dict["memory_id"] == "test-mem-1"
    manager.add_memory.assert_called_once()

def test_batch_add_tool():
    manager = MagicMock()
    manager.batch_add_memories.return_value = ["mem-1", "mem-2"]
    
    tools = CollivindTools(manager)
    args = {
        "memories": [
            {
                "content": "First",
                "summary": "Sum 1",
                "category": "fact"
            },
            {
                "content": "Second",
                "summary": "Sum 2",
                "category": "fact"
            }
        ]
    }
    
    response = tools.handle_call("collivind_batch_add", args)
    res_dict = json.loads(response)
    
    assert res_dict["status"] == "success"
    assert "mem-1" in res_dict["memory_ids"]
    manager.batch_add_memories.assert_called_once()
