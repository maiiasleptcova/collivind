from unittest.mock import MagicMock, patch

from collivind.mcp.server import MCPServer


@patch("collivind.mcp.server.create_all_backends")
@patch("collivind.mcp.server.load_config")
def test_server_initialize(mock_load, mock_backends):
    mock_backends.return_value = (MagicMock(), MagicMock(), MagicMock())
    server = MCPServer()

    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

    response = server.handle_request(req)

    assert response["id"] == 1
    assert "capabilities" in response["result"]
    assert server.initialized is True


@patch("collivind.mcp.server.create_all_backends")
@patch("collivind.mcp.server.load_config")
def test_server_tools_list(mock_load, mock_backends):
    mock_backends.return_value = (MagicMock(), MagicMock(), MagicMock())
    server = MCPServer()

    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

    response = server.handle_request(req)

    assert response["id"] == 2
    assert "tools" in response["result"]
    assert len(response["result"]["tools"]) == 13


@patch("collivind.mcp.server.create_all_backends")
@patch("collivind.mcp.server.load_config")
def test_degraded_server_still_lists_tools(mock_load, mock_backends):
    mock_backends.side_effect = RuntimeError("qdrant down")
    server = MCPServer()
    assert server.backends_available is False

    response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = [t["name"] for t in response["result"]["tools"]]
    assert "collivind_add_memory" in names
    assert len(names) == 13

    call = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "collivind_add_memory", "arguments": {}},
        }
    )
    assert call["result"]["isError"] is True
