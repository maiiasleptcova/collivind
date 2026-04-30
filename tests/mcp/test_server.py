import json
import pytest
from unittest.mock import MagicMock, patch
from collivind.mcp.server import MCPServer

@patch('collivind.mcp.server.load_config')
@patch('collivind.mcp.server.QdrantVectorStore')
@patch('collivind.mcp.server.Neo4jGraphStore')
@patch('collivind.mcp.server.HttpEmbeddingProvider')
def test_server_initialize(mock_http, mock_neo, mock_qdrant, mock_load):
    server = MCPServer()
    
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }
    
    response = server.handle_request(req)
    
    assert response["id"] == 1
    assert "capabilities" in response["result"]
    assert server.initialized is True

@patch('collivind.mcp.server.load_config')
@patch('collivind.mcp.server.QdrantVectorStore')
@patch('collivind.mcp.server.Neo4jGraphStore')
@patch('collivind.mcp.server.HttpEmbeddingProvider')
def test_server_tools_list(mock_http, mock_neo, mock_qdrant, mock_load):
    server = MCPServer()
    
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    response = server.handle_request(req)
    
    assert response["id"] == 2
    assert "tools" in response["result"]
    assert len(response["result"]["tools"]) == 9
