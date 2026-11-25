"""Reranker plumbing: the daemon's rerank op, and the providers' never-raise fallbacks.

Fake models only; no test loads a cross-encoder or spawns a process.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from test_embed_daemon import MODEL, FakeModel, daemon_on, data_dir, stop  # noqa: F401

from collivind.config import EmbeddingsConfig
from collivind.storage import embed_daemon
from collivind.storage.embedding_local import LocalEmbeddingProvider
from collivind.storage.embedding_service import HttpEmbeddingProvider

RERANK = "fake/cross-encoder"


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return [float(len(text)) for _, text in pairs]


def start(data_dir, rerank_model=RERANK, load_reranker=lambda name: FakeCrossEncoder()):  # noqa: F811
    path = embed_daemon.socket_path_for(data_dir)
    t = threading.Thread(
        target=embed_daemon.serve,
        args=(path, MODEL, 30.0),
        kwargs={"load_model": lambda n: FakeModel(), "rerank_model": rerank_model, "load_reranker": load_reranker},
        daemon=True,
    )
    t.start()
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()
    return path, t


def provider(data_dir, rerank_model=RERANK):  # noqa: F811
    return LocalEmbeddingProvider(EmbeddingsConfig(model=MODEL), data_dir=str(data_dir), rerank_model=rerank_model)


def test_daemon_reranks_and_provider_never_loads_in_process(data_dir, daemon_on):  # noqa: F811
    path, t = start(data_dir)
    try:
        assert embed_daemon.request(path, {"op": "info"})["rerank_model"] == RERANK
        p = provider(data_dir)
        with patch.object(embed_daemon, "load_cross_encoder", side_effect=AssertionError("loaded in-process")):
            assert p.rerank("q", ["a", "bbb"]) == [1.0, 3.0]
            assert p.embed("ab") == [2.0, 1.0]  # embeddings unaffected
    finally:
        stop(path, t)


def test_reranker_that_fails_to_load_leaves_embeddings_up(data_dir, daemon_on):  # noqa: F811
    path, t = start(data_dir, load_reranker=lambda name: (_ for _ in ()).throw(OSError("no model")))
    try:
        p = provider(data_dir)
        assert p.rerank("q", ["a"]) is None
        assert p.embed("ab") == [2.0, 1.0]
    finally:
        stop(path, t)


def test_daemon_without_a_reranker_is_replaced_on_the_first_rerank_request(data_dir, daemon_on):  # noqa: F811
    """Started by a rerank-off process; embeds keep resetting its idle timer, so it must not linger."""
    path, t = start(data_dir, rerank_model="")
    try:
        with patch.object(embed_daemon, "spawn") as spawn:
            assert provider(data_dir).rerank("q", ["a"]) is None
        spawn.assert_called_once()
        assert spawn.call_args.args[3] == RERANK
        t.join(5)
        assert not t.is_alive()
    finally:
        stop(path, t)


def test_daemon_with_another_reranker_is_not_killed(data_dir, daemon_on):  # noqa: F811
    """Two configs naming different models must not restart each other's daemon in a loop."""
    path, t = start(data_dir, rerank_model="other/cross-encoder")
    try:
        p = provider(data_dir)
        with patch.object(embed_daemon, "spawn") as spawn:
            assert p.rerank("q", ["a"]) is None
            assert p.embed("ab") == [2.0, 1.0]
        spawn.assert_not_called()
        assert t.is_alive()
        assert p.can_rerank is False  # backs off instead of building pools for nothing
    finally:
        stop(path, t)


def test_hook_without_any_daemon_skips_the_pool(data_dir, monkeypatch):  # noqa: F811
    monkeypatch.setenv(embed_daemon.ENV_DISABLE, "0")
    p = provider(data_dir)
    p.allow_in_process_rerank = False
    assert p.can_rerank is False


def test_hook_processes_never_load_a_cross_encoder(data_dir, monkeypatch):  # noqa: F811
    monkeypatch.setenv(embed_daemon.ENV_DISABLE, "0")
    p = provider(data_dir)
    p.allow_in_process_rerank = False
    with patch.object(embed_daemon, "load_cross_encoder", side_effect=AssertionError("loaded in a hook")):
        assert p.rerank("q", ["a"]) is None


def test_hook_manager_disallows_in_process_rerank():
    from collivind.cli.commands import hook

    manager = MagicMock()
    manager.embedding_provider = LocalEmbeddingProvider(EmbeddingsConfig(), rerank_model=RERANK)
    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        assert hook._load_manager().embedding_provider.allow_in_process_rerank is False


def test_no_daemon_means_no_rerank_and_no_in_process_load(data_dir, daemon_on):  # noqa: F811
    p = provider(data_dir)
    with (
        patch.object(embed_daemon, "spawn") as spawn,
        patch.object(embed_daemon, "load_cross_encoder", side_effect=AssertionError("loaded in-process")),
    ):
        assert p.rerank("q", ["a"]) is None
    spawn.assert_called_once()


def test_daemon_off_reranks_in_process(data_dir, monkeypatch):  # noqa: F811
    monkeypatch.setenv(embed_daemon.ENV_DISABLE, "0")
    p = provider(data_dir)
    with patch.object(embed_daemon, "load_cross_encoder", return_value=FakeCrossEncoder()) as load:
        assert p.rerank("q", ["ab"]) == [2.0]
        assert p.rerank("q", ["abc"]) == [3.0]
    load.assert_called_once_with(RERANK)


def test_in_process_load_failure_is_tried_once_then_rerank_is_off(data_dir, monkeypatch):  # noqa: F811
    monkeypatch.setenv(embed_daemon.ENV_DISABLE, "0")
    p = provider(data_dir)
    with patch.object(embed_daemon, "load_cross_encoder", side_effect=OSError("offline")) as load:
        assert p.rerank("q", ["a"]) is None
        assert p.can_rerank is False
        assert p.rerank("q", ["a"]) is None
    load.assert_called_once()


def test_rerank_off_is_never_asked(data_dir):  # noqa: F811
    p = provider(data_dir, rerank_model="")
    assert p.can_rerank is False
    assert p.rerank("q", ["a"]) is None


def _http(handler, rerank=True):
    p = HttpEmbeddingProvider(EmbeddingsConfig(service_url="http://embeddings"), rerank=rerank)
    p.client = httpx.Client(base_url="http://embeddings", transport=httpx.MockTransport(handler))
    return p


def test_http_rerank_round_trip():
    p = _http(lambda req: httpx.Response(200, json={"scores": [0.5, -1]}))
    assert p.can_rerank and p.rerank("q", ["a", "b"]) == [0.5, -1.0]


def test_http_old_image_without_rerank_endpoint_turns_rerank_off():
    calls = []
    p = _http(lambda req: calls.append(req) or httpx.Response(404))
    assert p.rerank("q", ["a"]) is None
    assert p.can_rerank is False
    assert p.rerank("q", ["a"]) is None
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, json={"scores": [1.0]}),  # wrong length
        httpx.Response(200, text="not json"),
    ],
)
def test_http_rerank_problems_return_none(response):
    p = _http(lambda req: response)
    assert p.rerank("q", ["a", "b"]) is None


def test_http_rerank_timeout_returns_none_without_retry_then_backs_off():
    handler = MagicMock(side_effect=httpx.ReadTimeout("slow"))
    p = _http(handler)
    assert p.rerank("q", ["a"]) is None
    assert p.can_rerank is False and p.rerank("q", ["a"]) is None  # a slow service is not asked on every prompt
    assert handler.call_count == 1


def test_http_rerank_off():
    p = _http(lambda req: pytest.fail("called"), rerank=False)
    assert p.can_rerank is False and p.rerank("q", ["a"]) is None
