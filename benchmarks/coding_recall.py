"""Coding-domain recall check: log the recall path's ranked list for coding prompts.

coding_recall_check.json is a synthetic store written for this check (several
projects of decisions, errors and fixes, architecture and status notes) plus
coding prompts: some with known relevant memories, some the store cannot help
with. Every memory goes through MemoryManager.add_memory into a temp
embedded store; every prompt is searched in its project at the hook's limit
with the cutoff lowered to the floor, and the whole reranked list is logged in
recall_log.py's format, so recall_sweep.py reads it ("category" 5 = no
relevant memory). With --hook, it also prints what the real get_context
returns with the shipped defaults (memories, tokens, relevant kept, empty).

    uv run python benchmarks/coding_recall.py --out benchmarks/data/recall_log_coding.jsonl [--hook]

Never touches ~/.collivind.
"""

import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomo_bench import live_end, make_manager  # noqa: E402
from recall_log import FLOOR, capture_rerank  # noqa: E402

DATA = Path(__file__).resolve().parent / "coding_recall_check.json"
LIMIT = 20
META = re.compile(r"^\[[A-Z]+\] \(score: ", re.M)  # one per memory get_context prints


def build(manager, data: dict) -> dict:
    """Ingest; returns live memory id -> the dataset ids it stands for."""
    from collivind.models import MemoryCategory, MemoryCreate

    owners = defaultdict(set)
    for m in data["memories"]:
        node = manager.add_memory(
            MemoryCreate(
                content=m["content"],
                summary=m["content"][:200],
                category=MemoryCategory(m["category"]),
                project_id=m["project"],
            )
        )
        owners[node.id].add(m["id"])
    resolved = defaultdict(set)
    for mid, ids in owners.items():
        resolved[live_end(manager.graph_store, mid)] |= ids
    return resolved


def hook_check(manager, data: dict, owners: dict) -> dict:
    """The shipped get_context, as the hook calls it: memories, tokens, relevant kept, empty."""
    from collivind.models import SearchQuery

    hooks, out = manager.config.hooks, defaultdict(list)
    for p in data["prompts"]:
        ctx = manager.get_context(
            p["prompt"], project_id=p["project"], limit=hooks.recall_limit, max_tokens=hooks.recall_max_tokens
        )
        empty = "No relevant context found" in ctx
        hits = [] if empty else manager.search(
            SearchQuery(query=p["prompt"], project_id=p["project"], limit=hooks.recall_limit,
                        min_rerank_score=manager.config.search.recall_min_score)
        )  # fmt: skip
        n = 0 if empty else len(META.findall(ctx))
        kept = {i for h in hits[:n] for i in owners.get(h.memory.id, ())}
        kind = "relevant" if p["relevant"] else "none"
        out[kind].append({"n": n, "tokens": 0 if empty else len(ctx) // 4, "empty": empty,
                          "kept": bool(set(p["relevant"]) & kept)})  # fmt: skip
    return {
        kind: {
            "prompts": len(rows),
            "mean_memories": round(sum(r["n"] for r in rows) / len(rows), 2),
            "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
            "empty": sum(r["empty"] for r in rows),
            "relevant_kept": sum(r["kept"] for r in rows),
        }
        for kind, rows in out.items()
    }


def main() -> None:
    from collivind.models import SearchQuery

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--prompts", default=None, help="prompt file to use instead of the store file's own prompts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hook", action="store_true", help="also report the shipped get_context defaults")
    ap.add_argument("--search", action="append", default=[], metavar="KEY=VALUE", help="[search] override")
    args = ap.parse_args()
    from locomo_bench import parse_search_overrides

    data = json.loads(Path(args.data).read_text())
    if args.prompts:  # another prompt set against the same store (coding_recall_confirm.json)
        data = {**data, "prompts": json.loads(Path(args.prompts).read_text())["prompts"]}
    with tempfile.TemporaryDirectory(prefix="collivind-coding-") as tmp:
        manager = make_manager(tmp, "BAAI/bge-small-en-v1.5", parse_search_overrides(args.search))
        owners = build(manager, data)
        print(f"{len(data['memories'])} memories -> {len(owners)} live", flush=True)
        if args.out:
            seen = capture_rerank(manager)
            with open(args.out, "w") as out:
                for p in data["prompts"]:
                    seen.clear()
                    hits = manager.search(
                        SearchQuery(query=p["prompt"], project_id=p["project"], limit=LIMIT, min_rerank_score=FLOOR)
                    )
                    full = seen.get("full") or hits
                    ranked = [
                        {"id": h.memory.id[:8], "p": h.rerank_score or 0.0, "s": round(h.score, 6),
                         "v": round(h.vector_score, 4),
                         "chars": len(h.memory.content), "summ": False,
                         "gold": bool(set(p["relevant"]) & owners.get(h.memory.id, set()))}
                        for h in full
                    ]  # fmt: skip
                    row = {"sid": p["project"], "qa_index": p["id"], "category": 1 if p["relevant"] else 5,
                           "store": "turns", "n_gold": len(p["relevant"]), "ranked": ranked}  # fmt: skip
                    out.write(json.dumps(row) + "\n")
        if args.hook:
            print(json.dumps(hook_check(manager, data, owners), indent=1))


if __name__ == "__main__":
    main()
