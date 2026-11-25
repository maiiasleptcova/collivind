"""Scorer for benchmarks/locomo_bench.py: no dataset, no model."""

import importlib.util
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "locomo_bench.py"
_spec = importlib.util.spec_from_file_location("locomo_bench", _BENCH)
assert _spec and _spec.loader
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_parse_evidence_splits_and_dedupes():
    assert bench.parse_evidence(["D1:3", "D8:6; D9:17", "D1:3", "junk"]) == ["D1:3", "D8:6", "D9:17"]


def test_hit_and_recall_at_k():
    ranked = [{"D1:1"}, {"D2:5", "D2:6"}, set(), {"D3:1"}]
    scores = bench.score_retrieval(ranked, {"D2:6", "D3:1"}, [1, 2, 5])
    assert scores == {
        "hit@1": 0.0,
        "recall@1": 0.0,
        "hit@2": 1.0,
        "recall@2": 0.5,
        "hit@5": 1.0,
        "recall@5": 1.0,
    }


def test_empty_retrieval_scores_zero():
    assert bench.score_retrieval([], {"D1:1"}, [5]) == {"hit@5": 0.0, "recall@5": 0.0}


def test_no_evidence_is_an_error_not_a_miss():
    with pytest.raises(ValueError):
        bench.score_retrieval([{"D1:1"}], set(), [5])


def test_summarise_keeps_adversarial_out_of_overall():
    rows = [
        {"category": 1, "category_name": "multi-hop", "hit@5": 1.0},
        {"category": 4, "category_name": "single-hop", "hit@5": 0.0},
        {"category": 5, "category_name": "adversarial", "hit@5": 1.0},
    ]
    summary = bench.summarise(rows, ["hit@5"])
    assert summary["overall"] == {"n": 2.0, "hit@5": 0.5}
    assert summary["all"]["n"] == 3.0
    assert summary["adversarial"] == {"n": 1.0, "hit@5": 1.0}


class _StubClient:
    """Stands in for anthropic.Anthropic: answers, then judges CORRECT."""

    def __init__(self):
        self.prompts = []
        self.messages = self

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        text = "CORRECT" if len(self.prompts) % 2 == 0 else "7 May 2023"

        class _Block:
            type = "text"

        block = _Block()
        block.text = text
        return type("Resp", (), {"content": [block]})()


def test_qa_judge_answers_then_grades():
    sample = {"conversation": {"speaker_a": "Caroline", "speaker_b": "Melanie"}}
    qa = {"question": "When?", "answer": "7 May 2023"}
    client = _StubClient()
    out = bench.qa_judge(client, "m", sample, qa, ["[8 May 2023] Caroline: went yesterday"])
    assert out == {"answer": "7 May 2023", "correct": 1.0}
    assert "went yesterday" in client.prompts[0] and "Gold answer: 7 May 2023" in client.prompts[1]


def test_parse_evidence_strips_zero_padding():
    assert bench.parse_evidence(["D30:05"]) == ["D30:5"]


def test_strict_hit_needs_the_evidence_text_itself():
    texts = ["[d] A: later near-duplicate", "[d] B: other", "[d] A: the evidence turn"]
    assert bench.score_strict(texts, {"[d] A: the evidence turn"}, [1, 2, 3]) == {
        "strict_hit@1": 0.0,
        "strict_hit@2": 0.0,
        "strict_hit@3": 1.0,
    }


def test_search_overrides_parse_bools_and_ints():
    assert bench.parse_search_overrides(["rerank=false", "rerank_pool=30", "rerank_model=x/y"]) == {
        "rerank": False,
        "rerank_pool": 30,
        "rerank_model": "x/y",
    }
