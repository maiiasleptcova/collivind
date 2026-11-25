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
    # deep enough to reach the dense tail and graph-expansion-only candidates
    cases += [{"query": q, "limit": 50} for q in queries[:8]]
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


# Scores move by up to ~1e-7 across CPU architectures (float32 vectors; the
# golden came from Apple silicon, CI runs on x86-64), so scores are compared
# within TOL. Ids are compared in exact order, except that neighbours whose
# golden scores are within TOL of each other form a tie group compared as a set:
# a platform may legitimately order a near-tie either way.
TOL = 1e-6


def _groups(ranked):
    groups, current = [], [ranked[0]] if ranked else []
    for prev, row in zip(ranked, ranked[1:]):
        if abs(prev[1] - row[1]) < TOL:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    return groups + ([current] if current else [])


def assert_same_ranking(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        label = f"{want['query']!r} limit={want['limit']}"
        assert {k: v for k, v in got.items() if k != "ranked"} == {k: v for k, v in want.items() if k != "ranked"}
        assert len(got["ranked"]) == len(want["ranked"]), label
        by_id = {row[0]: row for row in got["ranked"]}
        start = 0
        for group in _groups(want["ranked"]):
            window = got["ranked"][start : start + len(group)]
            assert {r[0] for r in window} == {r[0] for r in group}, f"{label}: order differs at rank {start}"
            for row in group:
                assert all(abs(a - b) <= TOL for a, b in zip(by_id[row[0]][1:], row[1:])), f"{label}: {row[0]} score"
            start += len(group)


def test_rerank_off_reproduces_0_18_0_ranking(tmp_path, frozen_now):
    expected = json.loads(GOLDEN.read_text())
    actual = _snapshot(str(tmp_path))
    assert any(r[3] > 0 for case in expected for r in case["ranked"]), "golden never exercised graph expansion"
    assert_same_ranking(actual, expected)


def test_ranking_comparison_catches_real_changes():
    want = [
        {
            "query": "q",
            "limit": 3,
            "ranked": [["a", 0.9, 0.9, 0.0], ["b", 0.5, 0.5, 0.0], ["c", 0.5000000004, 0.5, 0.0]],
        }
    ]
    swapped_tie = [{**want[0], "ranked": [want[0]["ranked"][0], want[0]["ranked"][2], want[0]["ranked"][1]]}]
    assert_same_ranking(swapped_tie, want)  # a near-tie may flip
    for broken in (
        [["b", 0.5, 0.5, 0.0], ["a", 0.9, 0.9, 0.0], ["c", 0.5, 0.5, 0.0]],  # real reorder
        [["a", 0.9, 0.9, 0.0], ["b", 0.5, 0.5, 0.0], ["d", 0.5, 0.5, 0.0]],  # different memory
        [["a", 0.9001, 0.9, 0.0], ["b", 0.5, 0.5, 0.0], ["c", 0.5, 0.5, 0.0]],  # score moved
    ):
        with pytest.raises(AssertionError):
            assert_same_ranking([{**want[0], "ranked": broken}], want)


if __name__ == "__main__":
    import tempfile

    search_engine_module.datetime = _Frozen  # type: ignore[misc]
    with tempfile.TemporaryDirectory() as tmp:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(_snapshot(tmp), indent=0) + "\n")
    print(f"wrote {GOLDEN}", file=sys.stderr)
