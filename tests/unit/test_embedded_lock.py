from unittest.mock import MagicMock, patch

import pytest

from collivind.config import QdrantConfig
from collivind.exceptions import CollivindError

LOCK_ERROR = RuntimeError("Storage folder /x/qdrant_data is already accessed by another instance of Qdrant client.")


def _make_store(client_side_effect, holders=None):
    with (
        patch("collivind.storage.qdrant_embedded.QdrantClient", side_effect=client_side_effect) as client_cls,
        patch("collivind.storage.qdrant_embedded.time.sleep") as sleep,
        patch.object(
            __import__("collivind.storage.qdrant_embedded", fromlist=["EmbeddedQdrantStore"]).EmbeddedQdrantStore,
            "_lock_holders",
            return_value=holders or [],
        ),
    ):
        from collivind.storage.qdrant_embedded import EmbeddedQdrantStore

        store = EmbeddedQdrantStore(data_dir="/tmp/x", config=QdrantConfig(), dimension=384)
        return store, client_cls, sleep


def test_lock_error_retries_then_reports_holder_pid():
    with pytest.raises(CollivindError) as err:
        _make_store(client_side_effect=LOCK_ERROR, holders=["12345"])
    message = str(err.value)
    assert "held by PID 12345" in message
    assert "kill 12345" in message
    assert "ONE process at a time" in message
    assert 'mode = "docker"' in message


def test_lock_error_without_lsof_still_explains():
    with pytest.raises(CollivindError) as err:
        _make_store(client_side_effect=LOCK_ERROR, holders=[])
    assert "lsof" in str(err.value)


def test_transient_lock_recovers_on_retry():
    client = MagicMock()
    store, client_cls, sleep = _make_store(client_side_effect=[LOCK_ERROR, client])
    assert store.client is client
    assert client_cls.call_count == 2
    sleep.assert_called_once()


def test_non_lock_error_raises_immediately():
    with pytest.raises(ValueError):
        _make_store(client_side_effect=ValueError("disk corrupt"))
