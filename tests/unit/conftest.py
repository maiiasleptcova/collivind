import pytest


@pytest.fixture(autouse=True)
def _no_external_probes(monkeypatch):
    """`collivind status` shells out to `claude mcp list` and `docker inspect`.
    Unit tests must not depend on either being installed on the host; tests
    that exercise those paths override these stubs."""
    monkeypatch.setattr("collivind.cli.commands.status._claude_mcp_listing", lambda: (None, "stubbed in tests"))
    monkeypatch.setattr("collivind.cli.commands.status._container_states", lambda names: [])
