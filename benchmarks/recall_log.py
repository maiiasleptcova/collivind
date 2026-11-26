"""Log the recall path's ranked list, with scores, for every LoCoMo question.

For each question of the chosen conversations, the whole reranked list the
recall path builds (before it is cut to `limit`) is recorded: per result its
cross-encoder score (sigmoid of the logit), combined score, text length,
whether it is a summary and whether its text is a gold evidence turn
(strict). No model is called and no dataset text is written: ids, numbers
and flags only, so stopping rules can be swept offline (recall_sweep.py).

    uv run python benchmarks/recall_log.py benchmarks/data/locomo10.json \\
        --convs conv-26,conv-30 --out benchmarks/data/recall_log_dev.jsonl \\
        [--summaries benchmarks/data/summaries.json]

The search runs at the hook's limit (20) with the cutoff lowered to 1e-30, so
every scored candidate is kept and an absolute cutoff can be re-applied
exactly. With --summaries the routed list (summary routing cuts to `limit`)
is logged after the turns-only list. Never touches ~/.collivind.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomo_bench import ingest, make_manager, parse_evidence, sessions_of, turn_text  # noqa: E402

FLOOR = 1e-30  # below sigmoid(-50), so nothing is dropped


def capture_rerank(manager) -> dict:
    """Wrap SearchEngine._rerank so the full list (before [:limit]) is kept."""
    engine = manager.search_engine
    seen: dict = {}
    original = engine._rerank

    def wrapped(*a, **kw):
        out = original(*a, **kw)
        seen["full"] = out
        return out

    engine._rerank = wrapped
    return seen


def row_of(hits, gold_texts) -> list:
    return [
        {
            "id": h.memory.id[:8],
            "p": h.rerank_score,
            "s": round(h.score, 6),
            "v": round(h.vector_score, 4),
            "chars": len(h.memory.content),
            "summ": h.memory.category == "summary",
            "gold": h.memory.content in gold_texts,
        }
        for h in hits
    ]


def main() -> None:
    from collivind.models import SearchQuery

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--convs", required=True, help="comma-separated sample ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--summaries", default=None)
    args = ap.parse_args()

    data = {s["sample_id"]: s for s in json.loads(Path(args.dataset).read_text())}
    summaries = json.loads(Path(args.summaries).read_text()) if args.summaries else {}
    with tempfile.TemporaryDirectory(prefix="collivind-recall-log-") as tmp, open(args.out, "w") as out:
        manager = make_manager(tmp, "BAAI/bge-small-en-v1.5")
        seen = capture_rerank(manager)
        for sid in args.convs.split(","):
            t0 = time.time()
            sample, pid = data[sid], f"locomo-{sid}"
            ingest(manager, sample, pid)
            texts = {t["dia_id"]: turn_text(t, w) for _, w, ts in sessions_of(sample["conversation"]) for t in ts}

            def log(store: str) -> None:
                for i, qa in enumerate(sample["qa"]):
                    gold = {texts[e] for e in parse_evidence(qa.get("evidence", [])) if e in texts}
                    seen.clear()
                    q = SearchQuery(query=qa["question"], project_id=pid, limit=args.limit, min_rerank_score=FLOOR)
                    hits = manager.search(q)
                    full = seen.get("full")
                    row = {
                        "sid": sid,
                        "qa_index": i,
                        "category": qa["category"],
                        "store": store,
                        "n_gold": len(gold),
                        # routed: what search returned; turns: the whole reranked list
                        "ranked": row_of(hits if store == "summ" or full is None else full, gold),
                    }
                    out.write(json.dumps(row) + "\n")

            log("turns")
            if summaries:
                from collivind.engine.summaries import Group, save_summary

                for number, when, _ in sessions_of(sample["conversation"]):
                    entry = summaries.get(f"{sid}|{number}")
                    if entry:
                        text = f"[{when}] Summary of the session on this date:\n{entry['summary']}"
                        save_summary(manager, pid, Group("session", f"{pid}-s{number}"), text)
                log("summ")
            print(f"{sid}: {len(sample['qa'])} questions, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
