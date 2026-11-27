"""Write-time dedup on hand-written coding-agent memory pairs (tests/fixtures/dedup_pairs.json).

Each pair is memory `a` then memory `b`, added one at a time with
MemoryManager.add_memory_checked into a temp embedded store. `kind` says
whether b restates a ("same") or is a different fact ("different"). The
outcome for b is one of:

    rejected  b was not stored; the call returned a
    merged    b was stored and a was retired (superseded)
    surfaced  both live, and a came back in b's possible_conflicts
    kept      both live, the agent was not told

A restatement is handled when it is rejected, merged or surfaced. A
different-fact pair loses a memory when it is rejected or merged.

Two layouts: "isolated" puts every pair in its own project (dedup and
conflicts are per project), "shared" puts all pairs in one project in order,
so template neighbours from other pairs can interfere.

    uv run --extra embedded python benchmarks/dedup_pairs_bench.py
    uv run --extra embedded python benchmarks/dedup_pairs_bench.py --search dedup_threshold=0.98 --out rows.jsonl

Never touches ~/.collivind: the store lives in a temp dir, embed daemon off.
"""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PAIRS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "dedup_pairs.json"
OUTCOMES = ("rejected", "merged", "surfaced", "kept")


def make_manager(data_dir: str, search: Dict[str, Any]):
    from collivind.config import CollivindConfig, EmbeddingsConfig, SearchConfig
    from collivind.engine.memory_manager import MemoryManager
    from collivind.storage.factory import create_all_backends

    config = CollivindConfig(
        data_dir=data_dir,
        mode="embedded",
        embeddings=EmbeddingsConfig(provider="local", model="BAAI/bge-small-en-v1.5", daemon=False),
        search=SearchConfig(**search),
    )
    return MemoryManager(*create_all_backends(config), config)


def to_create(mem: Dict[str, Any], project: str):
    from collivind.models import MemoryCategory, MemoryCreate

    return MemoryCreate(
        content=mem["content"],
        summary=mem["summary"],
        category=MemoryCategory(mem["category"]),
        project_id=project,
        tags=list(mem.get("tags", [])),
    )


def outcome(manager, a_id: str, b_id: str, conflict_ids: List[str]) -> str:
    if b_id == a_id:
        return "rejected"
    a = manager.graph_store.get_memory(a_id)
    if a is None or a.valid_to is not None or a.superseded_by:
        return "merged"
    return "surfaced" if a_id in conflict_ids else "kept"


def run(pairs: List[Dict[str, Any]], layout: str, search: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        manager = make_manager(tmp, search)
        for pair in pairs:
            project = "pairs" if layout == "shared" else pair["id"]
            a, _ = manager.add_memory_checked(to_create(pair["a"], project))
            b, conflicts = manager.add_memory_checked(to_create(pair["b"], project))
            rows.append(
                {
                    "id": pair["id"],
                    "kind": pair["kind"],
                    "group": pair["group"],
                    "hard": pair["hard"],
                    "layout": layout,
                    "outcome": outcome(manager, a.id, b.id, [c.memory.id for c in conflicts]),
                }
            )
        if layout == "shared":
            # a later pair can retire an earlier pair's memory: count that as a loss too
            live = {i for i, _, _ in manager.vector_store.iter_all()}
            ids = _ids_by_content(manager)
            for row, pair in zip(rows, pairs):
                row["all_live_at_end"] = all(ids.get(pair[k]["content"]) in live for k in ("a", "b"))
    return rows


def _ids_by_content(manager) -> Dict[str, str]:
    # the newest memory with that text: an exact resubmission keeps the first one's id
    return {m.content: m.id for m in manager.graph_store.iter_memories() if m.project_id == "pairs"}


def table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kind in ("same", "different"):
        sel = [r for r in rows if r["kind"] == kind]
        counts = Counter(r["outcome"] for r in sel)
        entry: Dict[str, Any] = {o: counts.get(o, 0) for o in OUTCOMES}
        entry["n"] = len(sel)
        if kind == "same":
            entry["handled_pct"] = round(100 * (len(sel) - counts.get("kept", 0)) / max(1, len(sel)), 1)
        else:
            lost = [r for r in sel if r["outcome"] in ("rejected", "merged") or r.get("all_live_at_end") is False]
            entry["lost"] = len(lost)
            entry["lost_pct"] = round(100 * len(lost) / max(1, len(sel)), 1)
        out[kind] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", default=str(PAIRS))
    parser.add_argument("--search", action="append", default=[], metavar="KEY=VALUE", help="SearchConfig override")
    parser.add_argument("--out", help="per-pair rows (JSONL)")
    args = parser.parse_args()
    search: Dict[str, Any] = {}
    for item in args.search:
        key, _, value = item.partition("=")
        search[key] = json.loads(value) if value[:1] in "-0123456789tf[{" else value

    pairs = json.loads(Path(args.pairs).read_text())
    rows = run(pairs, "isolated", search) + run(pairs, "shared", search)
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for layout in ("isolated", "shared"):
        print(layout, json.dumps(table([r for r in rows if r["layout"] == layout])))


if __name__ == "__main__":
    main()
