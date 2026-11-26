"""Score every system's LoCoMo and LongMemEval-S rankings with one set of definitions.

Inputs are ids-only ranking files in a systems directory (written by locomo_bench.py,
baselines.py, and an isolated MemPalace run; see BENCHMARKS.md). Nothing here reads
dataset text beyond question ids, categories and gold evidence ids.

    uv run python benchmarks/compare.py benchmarks/data/locomo10.json benchmarks/data/longmemeval_s.json \
        benchmarks/data/systems --out benchmarks/results_comparison.json

Definitions (identical for every system):
  turn hit@k      a gold evidence turn is among the top k units (systems whose unit is a turn)
  session hit@k   a gold evidence session is among the top k sessions, sessions ranked by their best unit
  session rec@k   fraction of the gold evidence sessions among the top k sessions
  chance          expected session hit@k for a random ranking of that conversation's sessions
LoCoMo: categories 1-4 with resolvable evidence. LongMemEval-S: all 500 questions, recall-any.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from locomo_bench import parse_evidence, sessions_of  # noqa: E402


def distinct(seq):
    return list(dict.fromkeys(seq))


def load_rows(path: Path):
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines()] if path.suffix == ".jsonl" else json.loads(text)


def locomo_gold(dataset):
    gold, n_sessions = {}, {}
    for s in json.loads(Path(dataset).read_text()):
        sess = sessions_of(s["conversation"])
        known = {t["dia_id"] for _, _, turns in sess for t in turns}
        n_sessions[s["sample_id"]] = len(sess)
        for i, qa in enumerate(s["qa"]):
            ev = set(parse_evidence(qa.get("evidence", []))) & known
            if qa["category"] in (1, 2, 3, 4) and ev:
                gold[(s["sample_id"], i)] = ev
    return gold, n_sessions


def chance(n, g, k):
    """P(at least one of g gold sessions in a random k of n)."""
    return 1.0 if k >= n else 1 - math.comb(n - g, k) / math.comb(n, k)


def sessions_ranked(row):
    if "ranked_sessions" in row and "ranked_units" not in row:
        return row["ranked_sessions"]  # the system's own session ranking
    if "retrieved_sessions" in row:
        return distinct(row["retrieved_sessions"])  # collivind: each memory's own session
    if "unit_sessions" in row:
        return distinct(row["unit_sessions"])
    return distinct(u.split(":")[0] for units in row["ranked_units"] for u in units)


def score_locomo(rows, gold, n_sessions, ks=(5, 10)):
    """rows: one list per k (a system whose ranking depends on the requested k), or one list for all."""
    out = {"n": 0}
    per_k = rows if isinstance(rows, dict) else {k: rows for k in ks}
    for k in ks:
        idx = {(r["sample_id"], r["qa_index"]): r for r in per_k[k]}
        turn, strict, shit, srec, floor = [], [], [], [], []
        for key, ev in gold.items():
            r = idx[key]
            gs = {e.split(":")[0] for e in ev}
            ranked_s = sessions_ranked(r)[:k]
            shit.append(bool(gs & set(ranked_s)))
            srec.append(len(gs & set(ranked_s)) / len(gs))
            floor.append(chance(n_sessions[key[0]], len(gs), k))
            units = r.get("ranked_units") or r.get("retrieved")
            if units is not None:
                turn.append(bool(ev & set().union(*map(set, units[:k]))) if units[:k] else False)
            if f"strict_hit@{k}" in r:
                strict.append(r[f"strict_hit@{k}"])
        out["n"] = len(gold)
        out[f"session_hit@{k}"] = 100 * sum(shit) / len(shit)
        out[f"session_recall@{k}"] = 100 * sum(srec) / len(srec)
        out[f"chance_session_hit@{k}"] = 100 * sum(floor) / len(floor)
        if turn:
            out[f"turn_hit@{k}"] = 100 * sum(turn) / len(turn)
        if strict:
            out[f"turn_hit_strict@{k}"] = 100 * sum(strict) / len(strict)
    return out


def score_lme(rows, gold, subset=None, ks=(5, 10)):
    """A depth is reported only if the file holds at least that many sessions for most questions."""
    idx = {r["question_id"]: r for r in rows}
    qids = [q for q in gold if (subset is None or q in subset) and q in idx]
    out = {"n": len(qids)}

    def ranked(q):
        r = idx[q]
        return r.get("ranked_sessions") or distinct(r.get("unit_sessions") or r.get("top_sessions") or [])

    for k in ks:
        if qids and sorted(len(ranked(q)) for q in qids)[len(qids) // 2] >= k:
            out[f"session_hit@{k}"] = 100 * sum(bool(set(ranked(q)[:k]) & gold[q]) for q in qids) / len(qids)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("locomo")
    p.add_argument("lme")
    p.add_argument("systems")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    d = Path(args.systems)
    gold, n_sessions = locomo_gold(args.locomo)
    lo = {
        "collivind 0.21.0 (default)": load_rows(d / "collivind_locomo_top50.jsonl"),
        "collivind, rerank off": load_rows(d / "collivind_norerank_locomo_top50.jsonl"),
        "dense bge-small, per turn": load_rows(d / "locomo_dense.json"),
        "BM25, per turn": load_rows(d / "locomo_bm25.json"),
        "MemPalace raw, per session (its --top-k 5 / 10 runs)": {
            5: load_rows(d / "mempalace_locomo_raw_top5run.json"),
            10: load_rows(d / "mempalace_locomo_raw_top10run.json"),
        },
        "MemPalace hybrid v5, per session (its --top-k 5 / 10 runs)": {
            5: load_rows(d / "mempalace_locomo_hybrid_top5run.json"),
            10: load_rows(d / "mempalace_locomo_hybrid_top10run.json"),
        },
        "MemPalace hybrid v5, per session (requested 50)": load_rows(d / "mempalace_locomo_hybrid.json"),
        "MemPalace raw, per session (requested 50)": load_rows(d / "mempalace_locomo_raw.json"),
        "MemPalace raw, per turn": load_rows(d / "mempalace_locomo_raw_turns.json"),
        "MemPalace hybrid v5, per turn": load_rows(d / "mempalace_locomo_hybrid_turns.json"),
    }  # fmt: skip
    report = {"locomo": {name: score_locomo(rows, gold, n_sessions) for name, rows in lo.items()}}

    lme_data = json.loads(Path(args.lme).read_text())
    lme_gold = {q["question_id"]: set(q["answer_session_ids"]) for q in lme_data}
    engine_rows = load_rows(HERE / "results_longmemeval_engine.jsonl")
    sample = {r["question_id"] for r in engine_rows}
    lme = {
        "dense bge-small, per turn": load_rows(d / "lme_dense.json"),
        "BM25, per turn": load_rows(d / "lme_bm25.json"),
        "MemPalace raw": load_rows(d / "mempalace_lme_raw.json"),
        "MemPalace hybrid v4": load_rows(d / "mempalace_lme_hybrid_v4.json"),
        "collivind raw vector store (bypasses the engine)": load_rows(
            Path("benchmarks/results_longmemeval_r5_bge.jsonl")
        ),
    }
    report["longmemeval_all"] = {name: score_lme(rows, lme_gold) for name, rows in lme.items()}
    report["longmemeval_sample"] = {name: score_lme(rows, lme_gold, sample) for name, rows in lme.items()}
    report["longmemeval_sample_ids"] = len(sample)
    for section in ("locomo", "longmemeval_all", "longmemeval_sample"):
        print(f"\n## {section}")
        cols = sorted(
            {k for v in report[section].values() for k in v},
            key=lambda c: (re.sub(r"\d+", "", c), int((re.findall(r"\d+", c) or [0])[0])),
        )
        print("| system | " + " | ".join(cols) + " |")
        print("|---|" + "---|" * len(cols))
        for name, v in report[section].items():
            print(
                f"| {name} | "
                + " | ".join(f"{v[c]:.1f}" if isinstance(v.get(c), float) else str(v.get(c, "")) for c in cols)
                + " |"
            )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
