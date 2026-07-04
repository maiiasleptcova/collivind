import shutil
import tempfile

from collivind.models.memory import MemoryCategory, MemoryCreate, MemoryNode, MemorySource
from collivind.storage.graph_sqlite import SqliteGraphStore


def _make_store():
    tmpdir = tempfile.mkdtemp()
    store = SqliteGraphStore(data_dir=tmpdir)
    store.initialize()
    return store, tmpdir


def test_new_memory_has_version_1():
    store, tmpdir = _make_store()
    try:
        mem = store.create_memory(MemoryCreate(
            content="v1", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))
        assert mem.version == 1
        assert mem.previous_version_id is None
    finally:
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_invalidate_sets_version_chain():
    store, tmpdir = _make_store()
    try:
        old = store.create_memory(MemoryCreate(
            content="v1", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))
        new = store.create_memory(MemoryCreate(
            content="v2", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))

        store.invalidate_memory(old.id, superseded_by=new.id, reason="updated")

        old_refreshed = store.get_memory(old.id)
        new_refreshed = store.get_memory(new.id)

        assert old_refreshed.valid_to is not None
        assert old_refreshed.superseded_by == new.id
        assert new_refreshed.version == 2
        assert new_refreshed.previous_version_id == old.id
    finally:
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_version_chain_returns_ordered_history():
    store, tmpdir = _make_store()
    try:
        v1 = store.create_memory(MemoryCreate(
            content="version 1", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))
        v2 = store.create_memory(MemoryCreate(
            content="version 2", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))
        v3 = store.create_memory(MemoryCreate(
            content="version 3", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))

        store.invalidate_memory(v1.id, superseded_by=v2.id, reason="update 1")
        store.invalidate_memory(v2.id, superseded_by=v3.id, reason="update 2")

        chain = store.get_version_chain(v1.id)
        assert len(chain) == 3
        assert chain[0].id == v1.id
        assert chain[1].id == v2.id
        assert chain[2].id == v3.id
    finally:
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_version_chain_from_middle():
    store, tmpdir = _make_store()
    try:
        v1 = store.create_memory(MemoryCreate(
            content="v1", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))
        v2 = store.create_memory(MemoryCreate(
            content="v2", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))
        v3 = store.create_memory(MemoryCreate(
            content="v3", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))

        store.invalidate_memory(v1.id, superseded_by=v2.id, reason="u1")
        store.invalidate_memory(v2.id, superseded_by=v3.id, reason="u2")

        chain = store.get_version_chain(v2.id)
        assert len(chain) == 3
        assert chain[0].id == v1.id
        assert chain[2].id == v3.id
    finally:
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_version_chain_single_memory():
    store, tmpdir = _make_store()
    try:
        mem = store.create_memory(MemoryCreate(
            content="solo", summary="s", category=MemoryCategory.FACT,
            project_id="p", user_id="u", source=MemorySource.MANUAL,
        ))
        chain = store.get_version_chain(mem.id)
        assert len(chain) == 1
        assert chain[0].id == mem.id
    finally:
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_version_chain_nonexistent():
    store, tmpdir = _make_store()
    try:
        chain = store.get_version_chain("nonexistent-id")
        assert chain == []
    finally:
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_to_dict_includes_version_fields():
    mem = MemoryNode(
        content="test", summary="s", category=MemoryCategory.FACT,
        version=3, previous_version_id="prev-id"
    )
    d = mem.to_dict()
    assert d["version"] == 3
    assert d["previous_version_id"] == "prev-id"
