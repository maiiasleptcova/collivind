"""Sweep recall stopping rules over a logged ranked list (recall_log.py, coding_recall.py).

No model calls. Each rule is applied the way the hook would see it: the
current cutoff (sigmoid score >= 0.00002) and the hook limit first, then the
rule, then get_context's token packing (len/4 per memory plus its header line).

    uv run python benchmarks/recall_sweep.py benchmarks/data/recall_log_dev.jsonl [--store turns]

Reported per rule, against the current default (cutoff 0.00002, limit 20,
2,400 tokens): memories and tokens injected per question, strict gold hit@5
and hit@10 inside what is injected, the share of the default's hit@10
questions the rule keeps, and for questions with no answer in the store
(LoCoMo category 5, coding prompts without a relevant memory) the memories
injected and the share that get nothing.
"""

import argparse
import json
import math
from typing import Callable, Dict, List

CUTOFF = 0.00002
META_CHARS = 22  # "[FACT] (score: 0.70)\n" + the trailing newline get_context adds
HEADER_TOKENS = len("--- Collivind Context ---") // 4


def logit(p: float) -> float:
    p = min(max(p, 1e-300), 1 - 1e-16)
    return math.log(p / (1 - p))


def pack(ranked: List[dict], budget: int) -> List[dict]:
    """get_context's greedy packing: stop at the first memory that does not fit (summaries are skipped)."""
    left, out = budget - HEADER_TOKENS, []
    for r in ranked:
        cost = (r["chars"] + META_CHARS) // 4
        if cost > left:
            if r.get("summ"):
                continue
            break
        left -= cost
        out.append(r)
    return out


Rule = Callable[[List[dict]], List[dict]]


def absolute(c: float) -> Rule:
    return lambda xs: [x for x in xs if x["p"] >= c]


def top(k: int) -> Rule:
    return lambda xs: xs[:k]


def relative(delta: float) -> Rule:
    """Keep results whose logit is within `delta` of the best one's."""

    def rule(xs):
        if not xs:
            return xs
        best = max(logit(x["p"]) for x in xs)
        return [x for x in xs if logit(x["p"]) >= best - delta]

    return rule


def ratio(f: float) -> Rule:
    """Keep results whose score is at least f x the best score."""

    def rule(xs):
        if not xs:
            return xs
        best = max(x["p"] for x in xs)
        return [x for x in xs if x["p"] >= f * best]

    return rule


def gap(min_keep: int = 1) -> Rule:
    """Cut at the largest logit drop between consecutive scores (sorted), keeping at least min_keep."""

    def rule(xs):
        if len(xs) <= min_keep:
            return xs
        ls = sorted((logit(x["p"]) for x in xs), reverse=True)
        drops = [(ls[i] - ls[i + 1], i) for i in range(min_keep - 1, len(ls) - 1)]
        _, i = max(drops)
        return [x for x in xs if logit(x["p"]) >= ls[i]]

    return rule


def chain(*rules: Rule) -> Rule:
    def rule(xs):
        for r in rules:
            xs = r(xs)
        return xs

    return rule


def regime(delta: float, cap: int, fb_k: int, fb_floor: float = CUTOFF, fb_cos: float = 0.0) -> Rule:
    """Two regimes, told apart by the best returned score (sigmoid 0.5 = the reranker's logit-0 boundary).

    Gate passed: keep results within `delta` logits of the best, at most `cap`.
    Gate failed (no result the reranker calls relevant; the order is the
    first stage's): keep at most `fb_k` results scoring at least `fb_floor`
    (and, if fb_cos, a dense cosine of at least fb_cos).
    """

    def rule(xs):
        if not xs:
            return xs
        if max(x["p"] for x in xs) >= 0.5:
            return relative(delta)(xs)[:cap]
        return [x for x in xs if x["p"] >= fb_floor and x.get("v", 1.0) >= fb_cos][:fb_k]

    return rule


def evaluate(rows: List[dict], rule: Rule, limit: int, budget: int) -> List[dict]:
    out = []
    for row in rows:
        base = [x for x in row["ranked"] if x["p"] >= CUTOFF][:limit]
        kept = pack(rule(base), budget)
        out.append(
            {
                "answerable": row["n_gold"] > 0 and row["category"] != 5,
                "n": len(kept),
                "tokens": HEADER_TOKENS + sum((x["chars"] + META_CHARS) // 4 for x in kept) if kept else 0,
                "hit5": any(x["gold"] for x in kept[:5]),
                "hit10": any(x["gold"] for x in kept[:10]),
                "hit": any(x["gold"] for x in kept),
            }
        )
    return out


def summarise(res: List[dict], base: List[dict]) -> Dict[str, float]:
    ans = [i for i, r in enumerate(res) if r["answerable"]]
    neg = [i for i, r in enumerate(res) if not r["answerable"]]
    base_hits = [i for i in ans if base[i]["hit10"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {
        "n": mean([res[i]["n"] for i in ans]),
        "tok": mean([res[i]["tokens"] for i in ans]),
        "hit5": 100 * mean([res[i]["hit5"] for i in ans]),
        "hit10": 100 * mean([res[i]["hit10"] for i in ans]),
        "hit_any": 100 * mean([res[i]["hit"] for i in ans]),
        "kept_of_base": 100 * mean([res[i]["hit10"] for i in base_hits]),
        "any_of_base": 100 * mean([res[i]["hit"] for i in ans if base[i]["hit"]]),
        "empty_ans": 100 * mean([res[i]["n"] == 0 for i in ans]),
        "neg_n": mean([res[i]["n"] for i in neg]),
        "neg_empty": 100 * mean([res[i]["n"] == 0 for i in neg]),
    }


RULES: Dict[str, Rule] = {
    "default (cutoff 2e-5)": lambda xs: xs,
    **{f"abs {c:g}": absolute(c) for c in (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.5)},
    **{f"top {k}": top(k) for k in (5, 8, 10, 12)},
    **{f"rel logit -{d:g}": relative(d) for d in (2, 3, 4, 5, 6, 7, 8)},
    **{f"ratio {f:g}": ratio(f) for f in (0.01, 0.05, 0.1, 0.2, 0.3)},
    "gap": gap(1),
    "gap keep>=3": gap(3),
    **{
        f"abs {c:g} + rel -{d:g}": chain(absolute(c), relative(d))
        for c in (1e-3, 1e-2, 0.05, 0.1)
        for d in (3, 4, 5, 6)
    },
    **{
        f"abs {c:g} + rel -{d:g} + top {k}": chain(absolute(c), relative(d), top(k))
        for c in (1e-3, 0.01, 0.05)
        for d in (4, 5, 6)
        for k in (8, 10, 12)
    },
    **{
        f"regime rel -{d:g} cap {c} | fb {k} >= {f:g}": regime(d, c, k, f)
        for d in (4, 6, 8, 10, 12, 99)
        for c in (8, 10, 12, 20)
        for k in (0, 1, 2, 3, 5)
        for f in (CUTOFF, 1e-3, 1e-2, 0.05)
        if not (k == 0 and f != CUTOFF)
    },
    **{
        f"regime rel -{d:g} cap {c} | fb {k} cos>={v:g}": regime(d, c, k, CUTOFF, v)
        for d in (6, 8, 10)
        for c in (10, 12)
        for k in (1, 2, 3, 5)
        for v in (0.6, 0.65, 0.7, 0.75)
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--store", default="turns", help="turns or summ (rows logged with summaries in the store)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--json", default=None, help="also write the table as JSON")
    ap.add_argument("--rules", default="", help="only print rules matching this regex")
    args = ap.parse_args()
    rows = [r for r in map(json.loads, open(args.log)) if r["store"] == args.store]
    base = evaluate(rows, RULES["default (cutoff 2e-5)"], 20, 2400)
    b = summarise(base, base)
    table = {}
    print(f"{len(rows)} rows, store={args.store}, limit={args.limit}, budget={args.budget}")
    print(
        f"{'rule':42} {'mem':>5} {'cut%':>5} {'tok':>5} {'h@5':>5} {'h@10':>5} {'any':>5} "
        f"{'kept%':>6} {'any%':>5} {'empty':>5} {'negN':>5} {'neg0%':>5}"
    )
    import re

    for name, rule in RULES.items():
        s = summarise(evaluate(rows, rule, args.limit, args.budget), base)
        s["cut"] = 100 * (1 - s["n"] / b["n"])
        table[name] = s
        if args.rules and not re.search(args.rules, name):
            continue
        print(
            f"{name:42} {s['n']:5.1f} {s['cut']:5.0f} {s['tok']:5.0f} {s['hit5']:5.1f} {s['hit10']:5.1f} "
            f"{s['hit_any']:5.1f} {s['kept_of_base']:6.1f} {s['any_of_base']:5.1f} {s['empty_ans']:5.1f} "
            f"{s['neg_n']:5.1f} {s['neg_empty']:5.0f}"
        )
    if args.json:
        open(args.json, "w").write(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
