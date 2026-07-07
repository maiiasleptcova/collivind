"""LongMemEval session-retrieval benchmark for collivind.

Measures session-level Recall@K: for each question, haystack sessions are
ingested into an embedded collivind store (one project per question), the
question is used as the search query, and we check whether an evidence
session appears in the top-K retrieved sessions.

Mode measured here is "raw": batch-embedded turns + vector search with
session-level aggregation. No LLM, no API keys, no reranking.

Usage:
    uv run python benchmarks/longmemeval_bench.py benchmarks/data/longmemeval_s.json \
        [--questions 500] [--top-k 5] [--out benchmarks/results_longmemeval.jsonl]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collivind.config import EmbeddingsConfig, QdrantConfig  # noqa: E402
from collivind.storage.embedding_local import LocalEmbeddingProvider  # noqa: E402
from collivind.storage.qdrant_embedded import EmbeddedQdrantStore  # noqa: E402

EMBED_BATCH = 256


def make_embedder(model: str, dimension: int):
    return LocalEmbeddingProvider(EmbeddingsConfig(provider="local", model=model, dimension=dimension))


def turns_of(session):
    """Flatten a session (list of {role, content} turns) to embeddable texts."""
    texts = []
    for turn in session:
        content = (turn.get("content") or "").strip()
        if content:
            texts.append(f"{turn.get('role', 'user')}: {content}"[:2000])
    return texts


def evaluate_question(store, embedder, question, top_k, query_prefix="", doc_prefix=""):
    q_sessions = question["haystack_sessions"]
    q_session_ids = question["haystack_session_ids"]
    answer_ids = set(question["answer_session_ids"])

    # ingest: one vector per turn, payload carries its session id
    texts, payloads = [], []
    for sid, session in zip(q_session_ids, q_sessions):
        for text in turns_of(session):
            texts.append(doc_prefix + text)
            payloads.append({"session_id": sid})
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        vectors = embedder.embed_batch(batch)
        for j, vec in enumerate(vectors):
            store.upsert(i + j, vec, payloads[i + j])

    # retrieve: rank sessions by their best-scoring turn
    qvec = embedder.embed(query_prefix + question["question"])
    hits = store.search(vector=qvec, limit=top_k * 20, filters={})
    top_sessions = []
    for hit in hits:
        sid = hit["payload"]["session_id"]
        if sid not in top_sessions:
            top_sessions.append(sid)
        if len(top_sessions) >= top_k:
            break

    return {
        "question_id": question["question_id"],
        "question_type": question.get("question_type"),
        "hit": bool(answer_ids.intersection(top_sessions)),
        "top_sessions": top_sessions,
        "answer_sessions": sorted(answer_ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--questions", type=int, default=None, help="Limit question count")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument(
        "--query-prefix",
        default="Represent this sentence for searching relevant passages: ",
        help="Prefix for query texts (model-specific)",
    )
    parser.add_argument("--doc-prefix", default="", help="Prefix for document texts (model-specific)")
    parser.add_argument("--out", default="benchmarks/results_longmemeval.jsonl")
    args = parser.parse_args()

    data = json.loads(Path(args.dataset).read_text())
    if args.questions:
        data = data[: args.questions]

    embedder = make_embedder(args.model, args.dimension)
    results = []
    started = time.time()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        for n, question in enumerate(data, 1):
            # fresh store per question: LongMemEval haystacks are per-question
            with tempfile.TemporaryDirectory() as tmp:
                store = EmbeddedQdrantStore(
                    data_dir=tmp, config=QdrantConfig(provider="embedded"), dimension=embedder.dimension
                )
                store.initialize()
                try:
                    result = evaluate_question(
                        store, embedder, question, args.top_k, args.query_prefix, args.doc_prefix
                    )
                finally:
                    store.close()
            results.append(result)
            out.write(json.dumps(result) + "\n")
            out.flush()
            if n % 10 == 0 or n == len(data):
                recall = sum(r["hit"] for r in results) / len(results)
                elapsed = time.time() - started
                print(f"[{n}/{len(data)}] R@{args.top_k}={recall:.3f} ({elapsed:.0f}s)", flush=True)

    recall = sum(r["hit"] for r in results) / len(results)
    by_type = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["hit"])
    print(f"\nFINAL R@{args.top_k}: {recall:.4f} on {len(results)} questions")
    for qtype, hits in sorted(by_type.items()):
        print(f"  {qtype}: {sum(hits) / len(hits):.3f} ({len(hits)}q)")


if __name__ == "__main__":
    main()
