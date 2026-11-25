"""LongMemEval-S through the real MemoryManager, session- and turn-level.

longmemeval_bench.py measures raw vector search with session aggregation;
session R@5 there (and here) is saturated near 98%, so it cannot show a
ranking change. This adds a turn-level metric: the dataset marks the turns
that hold the answer (`has_answer`), and turn_hit@k asks whether one of them
is among the top-k memories, by text (a near-duplicate that replaced the
turn does not count).

Stratified sample: N questions per type (seed 0), abstention questions
excluded. Each question gets a fresh embedded store in a temp dir; every turn
goes through add_memory, the question through search. Both arms (rerank on
and off) search the same store.

    uv run python benchmarks/longmemeval_engine.py benchmarks/data/longmemeval_s.json \\
        --per-type 15 --out benchmarks/results_longmemeval_engine.jsonl

Never touches ~/.collivind: embedded mode, temp data_dir, embed daemon off.
Output rows hold ids and scores only.
"""

import argparse
import json
import random
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

POOL = 50  # session R@5 ranks sessions by their best memory among the top 50
KS = (5, 10)


def top_sessions(per_memory_sessions: Sequence[Sequence[str]], k: int) -> List[str]:
    out: List[str] = []
    for group in per_memory_sessions:
        for sid in group:
            if sid not in out:
                out.append(sid)
    return out[:k]


def turn_text(when: str, turn: Dict[str, Any]) -> str:
    return f"[{when}] {turn.get('role', 'user')}: {(turn.get('content') or '').strip()}"[:2000]


def pick(data: List[Dict[str, Any]], per_type: int) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in data:
        if not q["question_id"].endswith("_abs"):
            by_type[q["question_type"]].append(q)
    rng = random.Random(0)
    return [q for t in sorted(by_type) for q in rng.sample(by_type[t], min(per_type, len(by_type[t])))]


def score(question, owners, answer_texts, hits_pool, hits_k) -> Dict[str, float]:
    sessions = [sorted(owners.get(h.memory.id, ())) for h in hits_pool]
    row = {"session_r@5": float(bool(set(top_sessions(sessions, 5)) & set(question["answer_session_ids"])))}
    for k in KS:
        row[f"turn_hit@{k}"] = float(any(h.memory.content in answer_texts for h in hits_k[:k]))
    return row


def run_question(question, embedder, arms):
    from collivind.config import CollivindConfig, EmbeddingsConfig
    from collivind.engine.memory_manager import MemoryManager
    from collivind.engine.search_engine import SearchEngine
    from collivind.models import MemoryCategory, MemoryCreate, SearchQuery
    from collivind.storage.factory import create_all_backends

    with tempfile.TemporaryDirectory(prefix="collivind-lme-") as tmp:
        config = CollivindConfig(
            data_dir=tmp,
            mode="embedded",
            embeddings=EmbeddingsConfig(provider="local", model="BAAI/bge-small-en-v1.5", daemon=False),
        )
        vector, graph, emb = create_all_backends(config)
        embedder = embedder or emb  # one loaded model (and reranker) for the whole run
        manager = MemoryManager(vector, graph, embedder, config)
        owners: Dict[str, set] = defaultdict(set)
        answer_texts = set()
        turns = 0
        for sid, when, session in zip(
            question["haystack_session_ids"], question["haystack_dates"], question["haystack_sessions"]
        ):
            for turn in session:
                if not (turn.get("content") or "").strip():
                    continue
                text = turn_text(when, turn)
                node = manager.add_memory(
                    MemoryCreate(
                        content=text, summary=text, category=MemoryCategory.FACT, project_id="lme", session_id=sid
                    )
                )
                owners[node.id].add(sid)
                turns += 1
                if turn.get("has_answer"):
                    answer_texts.add(text)
        assert str(config.expanded_data_dir).startswith(tmp), "store is not the temp dir"
        resolved: Dict[str, set] = defaultdict(set)
        for mid, sids in owners.items():
            node, seen = graph.get_memory(mid), {mid}
            while node is not None and node.superseded_by and node.superseded_by not in seen:
                seen.add(node.superseded_by)
                mid, node = node.superseded_by, graph.get_memory(node.superseded_by)
            resolved[mid] |= sids
        row: Dict[str, Any] = {
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "turns": turns,
            "answer_turns": len(answer_texts),
        }
        for arm, overrides in arms.items():
            engine = SearchEngine(vector, graph, embedder, replace(config.search, **overrides))
            pool = engine.search(SearchQuery(query=question["question"], project_id="lme", limit=POOL))
            top = engine.search(SearchQuery(query=question["question"], project_id="lme", limit=max(KS)))
            row[arm] = score(question, resolved, answer_texts, pool, top)
        vector.close()
        graph.close()
    return row, embedder


def report(rows: List[Dict[str, Any]], arms: Sequence[str]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {"overall": rows}
    for r in rows:
        groups.setdefault(r["question_type"], []).append(r)
    # turn metrics only where the dataset marks an answer turn
    summary: Dict[str, Any] = {}
    for name, rs in groups.items():
        with_turns = [r for r in rs if r["answer_turns"]]
        summary[name] = {"n": len(rs), "n_with_answer_turns": len(with_turns)}
        for arm in arms:
            summary[name][arm] = {
                "session_r@5": round(100 * sum(r[arm]["session_r@5"] for r in rs) / len(rs), 1),
                **{
                    f"turn_hit@{k}": round(100 * sum(r[arm][f"turn_hit@{k}"] for r in with_turns) / len(with_turns), 1)
                    if with_turns
                    else None
                    for k in KS
                },
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--per-type", type=int, default=15)
    parser.add_argument("--out", default="benchmarks/data/results_longmemeval_engine.jsonl")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    arms = {"rerank": {}, "no_rerank": {"rerank": False}}
    picked = pick(json.loads(Path(args.dataset).read_text()), args.per_type)
    print(f"{len(picked)} questions", flush=True)
    rows, embedder, started = [], None, time.time()
    with Path(args.out).open("w") as out:
        for n, question in enumerate(picked, 1):
            row, embedder = run_question(question, embedder, arms)
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if n % 5 == 0:
                print(f"[{n}/{len(picked)}] {time.time() - started:.0f}s", flush=True)
    summary = {"questions": len(rows), "seconds": round(time.time() - started), "scores": report(rows, list(arms))}
    print(json.dumps(summary, indent=2))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
