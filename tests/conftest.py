import pytest


@pytest.fixture(autouse=True)
def _no_embed_daemon(monkeypatch):
    """Never spawn a resident embedding daemon from a test; the daemon's own
    tests opt back in."""
    monkeypatch.setenv("COLLIVIND_EMBED_DAEMON", "0")


@pytest.fixture(autouse=True)
def _no_live_copilot_home(monkeypatch):
    """COPILOT_HOME overrides Path.home() for Copilot paths; a value set on
    the developer's machine would point tests at live config."""
    monkeypatch.delenv("COPILOT_HOME", raising=False)
