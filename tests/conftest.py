import pytest


@pytest.fixture(autouse=True)
def _no_embed_daemon(monkeypatch):
    """Never spawn a resident embedding daemon from a test; the daemon's own
    tests opt back in."""
    monkeypatch.setenv("COLLIVIND_EMBED_DAEMON", "0")
