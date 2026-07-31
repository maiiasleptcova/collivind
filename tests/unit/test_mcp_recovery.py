"""Issue #24: the MCP server is long-lived and is usually started before the
backends it needs. Caching a boot-time failure for the life of the process
means a plugin loaded before `collivind docker up` stays broken until the
agent is restarted."""

from unittest.mock import patch

from collivind.mcp.server import MCPServer


def _call(server, name="collivind_status"):
    return server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": {}}}
    )


def _is_error(response):
    return response["result"].get("isError") is True


class TestBackendRecovery:
    def test_recovers_when_backends_come_up_after_start(self):
        """The reported case: server boots with Qdrant down, containers start,
        the next tool call must work rather than repeating the boot error."""
        with patch("collivind.mcp.server.create_all_backends", side_effect=RuntimeError("connection refused")):
            server = MCPServer()
        assert server.backends_available is False

        with (
            patch("collivind.mcp.server.create_all_backends", return_value=(object(), object(), object())),
            patch("collivind.mcp.server.MemoryManager"),
            patch("collivind.mcp.server.CollivindTools") as tools,
        ):
            tools.return_value.call_tool.return_value = {"mode": "docker"}
            response = _call(server)

        assert not _is_error(response), response
        assert server.backends_available is True

    def test_still_reports_clearly_while_backends_stay_down(self):
        """Recovery must not soften the failure for the call that hits it (#8)."""
        with patch("collivind.mcp.server.create_all_backends", side_effect=RuntimeError("connection refused")):
            server = MCPServer()
            response = _call(server)

        assert _is_error(response)
        assert "connection refused" in response["result"]["content"][0]["text"]

    def test_retry_error_is_the_current_one_not_the_stale_boot_error(self):
        """A changed failure must be reported as it is now, not as it was."""
        with patch("collivind.mcp.server.create_all_backends", side_effect=RuntimeError("boot-time failure")):
            server = MCPServer()

        with patch("collivind.mcp.server.create_all_backends", side_effect=RuntimeError("neo4j auth rejected")):
            response = _call(server)

        text = response["result"]["content"][0]["text"]
        assert "neo4j auth rejected" in text
        assert "boot-time failure" not in text

    def test_healthy_server_does_not_rebuild_backends_per_call(self):
        """Recovery is for the broken case; a working server must not pay a
        rebuild on every call."""
        with (
            patch("collivind.mcp.server.create_all_backends", return_value=(object(), object(), object())) as build,
            patch("collivind.mcp.server.MemoryManager"),
            patch("collivind.mcp.server.CollivindTools") as tools,
        ):
            tools.return_value.call_tool.return_value = {}
            server = MCPServer()
            _call(server)
            _call(server)
            assert build.call_count == 1
