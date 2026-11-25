"""With rerank off, search ranks exactly as 0.18.0 did.

The golden file was written by `_snapshot` against the untouched 0.18.0
engine (origin/main 8bc049c), before the hybrid/rerank change. Regenerate it
only on purpose: `uv run python tests/unit/test_rerank_off_golden.py`.
"""

import importlib.util
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from collivind.config import CollivindConfig
from collivind.engine import search_engine as search_engine_module
from collivind.engine.search_engine import SearchEngine
from collivind.models import SearchQuery

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "search_bench.py"
_spec = importlib.util.spec_from_file_location("search_bench", _BENCH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

GOLDEN = Path(__file__).with_name("data") / "golden_ranking_0_18_0.json"
MEMORIES = 1200
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


class _Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


def _cases(texts):
    queries = bench.make_queries(texts, 25)
    cases = [{"query": q, "limit": limit} for q in queries for limit in (5, 10)]
    cases += [{"query": q, "limit": 10, "entity_names": ["Ent 0", "Ent 3"]} for q in queries[:5]]
    cases += [{"query": q, "limit": 10, "include_graph": False} for q in queries[:5]]
    return cases


def _search_config():
    search = CollivindConfig().search
    # the switches only exist after the change; the snapshot predates them
    return replace(search, rerank=False) if hasattr(search, "rerank") else search


def _snapshot(data_dir):
    graph, vector, embedder, texts = bench.seed_stores(data_dir, MEMORIES)
    engine = SearchEngine(vector, graph, embedder, _search_config())
    out = []
    for case in _cases(texts):
        results = engine.search(SearchQuery(**case))
        out.append(
            {
                **case,
                "ranked": [
                    [r.memory.id, round(r.score, 9), round(r.vector_score, 9), round(r.graph_score, 9)] for r in results
                ],
            }
        )
    graph.close()
    vector.close()
    return out


@pytest.fixture
def frozen_now(monkeypatch):
    monkeypatch.setattr(search_engine_module, "datetime", _Frozen)


def test_rerank_off_reproduces_0_18_0_ranking(tmp_path, frozen_now):
    expected = json.loads(GOLDEN.read_text())
    actual = _snapshot(str(tmp_path))
    assert any(r[3] > 0 for case in expected for r in case["ranked"]), "golden never exercised graph expansion"
    assert actual == expected


if __name__ == "__main__":
    import tempfile

    search_engine_module.datetime = _Frozen  # type: ignore[misc]
    with tempfile.TemporaryDirectory() as tmp:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(_snapshot(tmp), indent=0) + "\n")
    print(f"wrote {GOLDEN}", file=sys.stderr)
