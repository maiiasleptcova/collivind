"""LoCoMo benchmark for collivind, run through the real MemoryManager.

Each LoCoMo conversation is ingested turn by turn with MemoryManager.add_memory
(dedup, contradiction detection, enrichment and all) into a fresh embedded-mode
store in a temp dir, one project per conversation. Every question then goes
through MemoryManager.search and is scored on whether the gold evidence turns
(LoCoMo's `evidence` dia_ids, e.g. "D1:3") come back in the top K.

Retrieval mode (default) needs no API key and no network beyond the local
embedding model. QA mode (--qa) additionally asks a Claude model to answer from
the retrieved turns and a second call to judge the answer against the gold one;
it needs the `anthropic` package and ANTHROPIC_API_KEY and costs money.

    uv run python benchmarks/locomo_bench.py benchmarks/data/locomo10.json --limit 50
    uv run python benchmarks/locomo_bench.py benchmarks/data/locomo10.json          # all 1,986 questions
    uv run python benchmarks/locomo_bench.py benchmarks/data/locomo10.json --qa --limit 20

Never touches ~/.collivind or the docker services: the config is built by hand
with mode="embedded", data_dir=<tempdir> and the embed daemon off.
"""

import argparse
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# LoCoMo QA categories. The dataset stores integers; the names follow the
# mapping used by the LoCoMo authors' eval code and later papers (see
# BENCHMARKS.md). 5 is adversarial: the question is unanswerable from the
# conversation, so it is reported separately and kept out of the headline.
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ADVERSARIAL = 5
DIA_ID = re.compile(r"D(\d+):(\d+)")


def parse_evidence(raw: Iterable[str]) -> List[str]:
    """Normalise LoCoMo evidence entries: a few hold several ids ("D8:6; D9:17"),
    one is zero-padded ("D30:05")."""
    ids: List[str] = []
    for entry in raw:
        for session, turn in DIA_ID.findall(str(entry)):
            found = f"D{int(session)}:{int(turn)}"
            if found not in ids:
                ids.append(found)
    return ids


def score_retrieval(ranked: Sequence[Set[str]], evidence: Set[str], ks: Sequence[int]) -> Dict[str, float]:
    """Score one question.

    ranked: per retrieved memory (best first), the dia_ids it stands for.
    hit@k: any evidence turn is in the top k memories.
    recall@k: fraction of the evidence turns in the top k memories.
    """
    if not evidence:
        raise ValueError("question has no resolvable evidence")
    scores: Dict[str, float] = {}
    for k in ks:
        found = set().union(*ranked[:k]) & evidence if ranked[:k] else set()
        scores[f"hit@{k}"] = float(bool(found))
        scores[f"recall@{k}"] = len(found) / len(evidence)
    return scores


def score_strict(texts: Sequence[str], evidence_texts: Set[str], ks: Sequence[int]) -> Dict[str, float]:
    """strict_hit@k: a top-k memory's text IS an evidence turn's text.

    hit@k credits a memory with every turn it superseded (near-duplicate
    dedup), even though the text returned is the later turn; this does not.
    """
    return {f"strict_hit@{k}": float(any(t in evidence_texts for t in texts[:k])) for k in ks}


def summarise(results: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Mean of each metric per category, plus 'overall' (non-adversarial) and 'all'."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[r["category_name"]].append(r)
        groups["all"].append(r)
        if r["category"] != ADVERSARIAL:
            groups["overall"].append(r)
    return {
        name: {"n": float(len(rows)), **{m: sum(row[m] for row in rows) / len(rows) for m in metrics}}
        for name, rows in groups.items()
    }


def turn_text(turn: Dict[str, Any], when: str) -> str:
    text = f"[{when}] {turn['speaker']}: {turn['text']}"
    if turn.get("blip_caption"):
        text += f" [shares a photo: {turn['blip_caption']}]"
    return text


def sessions_of(conversation: Dict[str, Any]) -> List[tuple]:
    """(session number, date string, turns) in session order."""
    out = []
    for key, turns in conversation.items():
        m = re.fullmatch(r"session_(\d+)", key)
        if m and isinstance(turns, list):
            out.append((int(m.group(1)), conversation.get(f"{key}_date_time", ""), turns))
    return sorted(out)


def make_manager(data_dir: str, model: str, search: Optional[Dict[str, Any]] = None):
    """search: SearchConfig overrides, e.g. {"rerank": False}."""
    from collivind.config import CollivindConfig, EmbeddingsConfig, SearchConfig
    from collivind.engine.memory_manager import MemoryManager
    from collivind.storage.factory import create_all_backends

    config = CollivindConfig(
        data_dir=data_dir,
        mode="embedded",
        embeddings=EmbeddingsConfig(provider="local", model=model, daemon=False),
        search=SearchConfig(**(search or {})),
    )
    vector, graph, embedder = create_all_backends(config)
    return MemoryManager(vector, graph, embedder, config)


def ingest(manager, sample: Dict[str, Any], project_id: str) -> Dict[str, Any]:
    """Add every turn as a memory. Returns the live memory_id -> dia_ids map and counts.

    Nothing benchmark-specific goes into the memory (no tags), so the embedded
    text is exactly what a user's turn would produce. An exact duplicate
    returns the existing memory, so its dia_id joins that memory's set. A
    near-duplicate supersedes an older memory; afterwards every memory's ids
    are carried along its superseded_by chain to the live end.
    """
    from collivind.models import MemoryCategory, MemoryCreate

    owners: Dict[str, Set[str]] = defaultdict(set)
    turns = 0
    for number, when, session in sessions_of(sample["conversation"]):
        for turn in session:
            text = turn_text(turn, when)
            node = manager.add_memory(
                MemoryCreate(
                    content=text,
                    summary=text,
                    category=MemoryCategory.FACT,
                    project_id=project_id,
                    session_id=f"{project_id}-s{number}",
                )
            )
            owners[node.id].add(turn["dia_id"])
            turns += 1
    live_ids = {i for i, _, payload in manager.vector_store.iter_all() if payload.get("project_id") == project_id}
    resolved: Dict[str, Set[str]] = defaultdict(set)
    for memory_id, dia_ids in owners.items():
        resolved[live_end(manager.graph_store, memory_id)] |= dia_ids
    reachable = set().union(*(resolved[i] for i in live_ids if i in resolved))
    return {
        "owners": resolved,
        "turns": turns,
        "live_memories": len(live_ids),
        "turns_reachable": len(reachable),
        "_reachable": reachable,
    }


def live_end(graph, memory_id: str) -> str:
    seen = {memory_id}
    node = graph.get_memory(memory_id)
    while node is not None and node.superseded_by and node.superseded_by not in seen:
        seen.add(node.superseded_by)
        memory_id = node.superseded_by
        node = graph.get_memory(memory_id)
    return memory_id


def select_questions(data: List[Dict[str, Any]], limit: Optional[int]) -> List[tuple]:
    """(sample, qa) pairs in dataset order, cut to `limit` questions."""
    picked = [(sample, qa) for sample in data for qa in sample["qa"]]
    return picked[:limit] if limit else picked


# --- optional LLM QA mode -------------------------------------------------

ANSWER_PROMPT = """You are answering a question about a long conversation between {a} and {b}.
Below are the conversation turns a memory system retrieved for this question, each with its session date.

{context}

Question: {question}
Answer in a short phrase, using only the turns above. If they do not contain the answer, say "unknown"."""

JUDGE_PROMPT = """Grade a short answer against the gold answer to a question.
Question: {question}
Gold answer: {gold}
Generated answer: {answer}
Count it CORRECT if it refers to the same fact as the gold answer, even if worded differently or
less detailed; dates count if they name the same day/period. Otherwise WRONG.
Reply with exactly one word: CORRECT or WRONG."""


def _ask(client, model: str, prompt: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def qa_judge(client, model: str, sample, qa, retrieved_texts: List[str]) -> Dict[str, Any]:
    conv = sample["conversation"]
    if "answer" not in qa:
        return {"answer": None, "correct": 0.0}
    answer = _ask(
        client,
        model,
        ANSWER_PROMPT.format(
            a=conv["speaker_a"], b=conv["speaker_b"], context="\n".join(retrieved_texts), question=qa["question"]
        ),
    )
    verdict = _ask(client, model, JUDGE_PROMPT.format(question=qa["question"], gold=qa["answer"], answer=answer))
    return {"answer": answer, "correct": float(verdict.upper().startswith("CORRECT"))}


def make_judge_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("--qa needs ANTHROPIC_API_KEY in the environment")
    try:
        import anthropic
    except ImportError:
        raise SystemExit("--qa needs the anthropic SDK: uv pip install anthropic") from None
    return anthropic.Anthropic()


# --- driver ---------------------------------------------------------------


def parse_search_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    """["rerank=false", "rerank_pool=30"] -> {"rerank": False, "rerank_pool": 30}."""
    out: Dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw.lower() if raw.lower() in ("true", "false") else raw)
        except json.JSONDecodeError:
            out[key] = raw  # a plain string, e.g. a model name
    return out


def run(args) -> None:
    from collivind.models import SearchQuery

    data = json.loads(Path(args.dataset).read_text())
    ks = sorted({int(k) for k in args.top_k.split(",")})
    pairs = select_questions(data, args.limit)
    client = make_judge_client() if args.qa else None
    metrics = [f"{m}@{k}" for k in ks for m in ("hit", "strict_hit", "recall")] + (["correct"] if args.qa else [])

    results: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = defaultdict(int)
    ingest_stats = []
    started = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qa_index = {id(qa): i for sample in data for i, qa in enumerate(sample["qa"])}
    with tempfile.TemporaryDirectory(prefix="collivind-locomo-") as tmp, out_path.open("w") as out:
        manager = make_manager(tmp, args.model, parse_search_overrides(args.search))
        print(f"search config: {manager.config.search}", flush=True)
        ingested: Dict[str, Dict[str, Any]] = {}
        for n, (sample, qa) in enumerate(pairs, 1):
            pid = f"locomo-{sample['sample_id']}"
            if pid not in ingested:
                t0 = time.time()
                ingested[pid] = ingest(manager, sample, pid)
                stats = {k: v for k, v in ingested[pid].items() if k not in ("owners", "_reachable")}
                ingest_stats.append({"project": pid, **stats, "seconds": round(time.time() - t0, 1)})
                print(f"ingested {pid}: {ingest_stats[-1]}", flush=True)
                if not any(Path(tmp).iterdir()):
                    raise SystemExit(f"nothing was written to {tmp}: the store is not the temp dir")
            turn_texts = {
                t["dia_id"]: turn_text(t, when) for _, when, s in sessions_of(sample["conversation"]) for t in s
            }
            known = set(turn_texts)
            evidence = set(parse_evidence(qa.get("evidence", [])))
            if not evidence:
                skipped["no evidence"] += 1
                continue
            if not evidence <= known:
                evidence &= known
                if not evidence:
                    skipped["only unresolvable evidence ids (dropped)"] += 1
                    continue
                skipped["some unresolvable evidence ids (scored on the rest)"] += 1
            if not evidence <= ingested[pid]["_reachable"]:
                skipped["evidence turn in no live memory (scored as a miss)"] += 1

            # the recall path (prompt hook / get_context): the recall cutoff applies
            hits = manager.search(
                SearchQuery(
                    query=qa["question"],
                    project_id=pid,
                    limit=max(ks),
                    min_rerank_score=manager.config.search.recall_min_score,
                )
            )
            ranked = [set(ingested[pid]["owners"].get(h.memory.id, set())) for h in hits]
            row: Dict[str, Any] = {
                # ids only: the dataset text is CC BY-NC and is not redistributed
                "sample_id": sample["sample_id"],
                "qa_index": qa_index[id(qa)],  # not .index(qa): LoCoMo repeats a few identical questions
                "category": qa["category"],
                "category_name": CATEGORIES.get(qa["category"], str(qa["category"])),
                "evidence": sorted(evidence),
                "retrieved": [sorted(r) for r in ranked],
                # the session each returned memory itself belongs to (not the turns it superseded)
                "retrieved_sessions": [f"D{(h.memory.session_id or '').rsplit('-s', 1)[-1]}" for h in hits],
                **score_retrieval(ranked, evidence, ks),
                **score_strict([h.memory.content for h in hits], {turn_texts[e] for e in evidence}, ks),
            }
            if client is not None and qa["category"] != ADVERSARIAL:
                row.update(qa_judge(client, args.judge_model, sample, qa, [h.memory.content for h in hits]))
            elif client is not None:
                row["correct"] = 0.0  # adversarial: no gold answer; excluded from headline anyway
            results.append(row)
            out.write(json.dumps(row) + "\n")
            if n % 100 == 0:
                print(f"[{n}/{len(pairs)}] {time.time() - started:.0f}s", flush=True)

    report = {
        "questions": len(results),
        "skipped": dict(skipped),
        "seconds": round(time.time() - started, 1),
        "ingest": ingest_stats,
        "scores": summarise(results, metrics),
    }
    print(json.dumps(report, indent=2))
    if args.summary:
        Path(args.summary).write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="path to locomo10.json")
    parser.add_argument("--limit", type=int, default=None, help="first N questions (smoke mode)")
    parser.add_argument("--top-k", default="5,10", help="comma-separated K values")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="local embedding model")
    parser.add_argument("--qa", action="store_true", help="also answer + LLM-judge each question (paid API)")
    parser.add_argument("--judge-model", default="claude-haiku-5-5")
    parser.add_argument(
        "--out",
        default=None,
        help="per-question JSONL (default: benchmarks/results_locomo.jsonl, or benchmarks/data/ with --limit)",
    )
    parser.add_argument("--summary", default=None, help="also write the summary JSON here")
    parser.add_argument(
        "--search",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="[search] override, repeatable, e.g. --search rerank=false (default: the shipped defaults)",
    )
    args = parser.parse_args()
    if args.out is None:
        # a smoke run must not overwrite the committed full-run results
        args.out = "benchmarks/data/results_locomo_smoke.jsonl" if args.limit else "benchmarks/results_locomo.jsonl"
    run(args)


if __name__ == "__main__":
    main()
