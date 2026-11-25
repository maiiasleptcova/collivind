# ruff: noqa: E501  (the reader and judge prompts are kept verbatim)
"""LoCoMo QA with and without summary memories, through the real MemoryManager.

Every turn of each conversation is added with MemoryManager.add_memory (one
session_id per LoCoMo session, as locomo_bench.py does). Each question is then
answered from:

  turns50   search(limit=50) with no summaries in the store (the baseline)
  routed    search(limit=28) after one summary per session was saved with
            engine.summaries.save_summary: the top 3 summaries, the best of
            their member turns and the best other turns
  hook_*    get_context(limit=20, max_tokens=800): what the prompt hook
            injects, before (hook_turns) and after (hook_summ) summaries

Summaries are agent-written input (for example the evaluation's summaries.json,
{"conv-26|1": {"when": ..., "summary": "- fact\\n- fact"}}): this script never
writes summary text. Sample = the 300 questions qa.py in the 2026-11-25
evaluation used (categories 1-4 with an answer and resolvable evidence, shuffled
with seed 0). Reader and judge run through the Claude CLI on the subscription,
isolated from user settings, hooks, plugins and MCP (no API key), with Mem0's
judge prompt; every call is cached by prompt hash so a run resumes.

    uv run python benchmarks/locomo_summary_qa.py locomo10.json summaries.json --work benchmarks/data/summ
    uv run python benchmarks/locomo_summary_qa.py locomo10.json summaries.json --work benchmarks/data/summ --qa

Everything written to --work holds dataset text (LoCoMo is CC BY-NC): keep it
out of commits (benchmarks/data/ is git-ignored). Never touches ~/.collivind.
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomo_bench import ADVERSARIAL, ingest, make_manager, parse_evidence, sessions_of  # noqa: E402

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
ARMS = ("turns50", "routed", "hook_turns", "hook_summ")

READER = """You are an intelligent memory assistant answering a question about a long conversation between {a} and {b}.

Below are conversation turns. Each starts with the date and time of the session it was said in.

# INSTRUCTIONS
1. Use only the turns below.
2. Pay attention to the timestamps. Convert relative time references ("last year", "two months ago", "yesterday") to specific dates, months or years using the timestamp of the turn that says them.
3. If turns contradict each other, prefer the most recent.
4. Answer in fewer than 6 words where possible. Give only the answer, no explanation.
5. If the turns do not contain the answer, give your best guess from them.

# TURNS
{context}

Question: {question}
Answer:"""

# Mem0's ACCURACY_PROMPT (mem0 v1.0.0 evaluation/metrics/llm_judge.py), verbatim
JUDGE = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a ’gold’ (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def sample_questions(data: List[dict], n: int) -> List[tuple]:
    """The 2026-11-25 evaluation's sample: (sample_id, qa_index), reproduced from the dataset alone."""
    picked = []
    for s in data:
        known = {t["dia_id"] for _, _, turns in sessions_of(s["conversation"]) for t in turns}
        for i, qa in enumerate(s["qa"]):
            if qa["category"] in CATEGORIES and "answer" in qa and set(parse_evidence(qa.get("evidence", []))) & known:
                picked.append((s["sample_id"], i))
    random.Random(0).shuffle(picked)
    return picked[:n]


def build_contexts(data: Dict[str, dict], summaries: dict, sample: List[tuple]) -> Dict[str, dict]:
    from collivind.engine.summaries import Group, save_summary
    from collivind.models import SearchQuery

    wanted = sorted({sid for sid, _ in sample})
    out: Dict[str, dict] = {f"{sid}:{i}": {} for sid, i in sample}
    stats = {"summaries": 0, "gold_summary_routed": 0, "routed_searches": 0}
    with tempfile.TemporaryDirectory(prefix="collivind-summ-") as tmp:
        manager = make_manager(tmp, "BAAI/bge-small-en-v1.5")
        cutoff = manager.config.search.recall_min_score

        def search(q, pid, limit):
            hits = manager.search(SearchQuery(query=q, project_id=pid, limit=limit, min_rerank_score=cutoff))
            return [h.memory.content for h in hits], hits

        for sid in wanted:
            pid = f"locomo-{sid}"
            t0 = time.time()
            ingest(manager, data[sid], pid)
            qs = [(i, data[sid]["qa"][i]["question"]) for s2, i in sample if s2 == sid]
            for i, q in qs:
                out[f"{sid}:{i}"]["turns50"] = search(q, pid, 50)[0]
                out[f"{sid}:{i}"]["hook_turns"] = [manager.get_context(q, project_id=pid, limit=20, max_tokens=800)]
            for number, when, _ in sessions_of(data[sid]["conversation"]):
                entry = summaries.get(f"{sid}|{number}")
                if not entry:
                    continue
                text = f"[{when}] Summary of the session on this date:\n{entry['summary']}"
                save_summary(manager, pid, Group("session", f"{pid}-s{number}"), text)
                stats["summaries"] += 1
            for i, q in qs:
                texts, hits = search(q, pid, 28)
                out[f"{sid}:{i}"]["routed"] = texts
                out[f"{sid}:{i}"]["hook_summ"] = [manager.get_context(q, project_id=pid, limit=20, max_tokens=800)]
                gold = {int(e[1:].split(":")[0]) for e in parse_evidence(data[sid]["qa"][i].get("evidence", []))}
                routed = {h.memory.session_id for h in hits if h.memory.category == "summary"}
                stats["routed_searches"] += bool(routed)
                stats["gold_summary_routed"] += bool({f"{pid}-s{g}" for g in gold} & routed)
            print(f"{sid}: {len(qs)} questions, {time.time() - t0:.0f}s", flush=True)
    stats["questions"] = len(sample)
    print(json.dumps(stats), flush=True)
    return out


class Claude:
    """`claude -p` isolated from the user's settings, hooks, plugins and MCP; cached by prompt hash."""

    def __init__(self, model: str, cache_path: Path):
        self.model, self.lock, self.cache = model, threading.Lock(), {}
        self.empty = Path(tempfile.mkdtemp(prefix="collivind-claude-"))
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                row = json.loads(line)
                self.cache.setdefault(row["k"], row["v"])  # first answer wins, as it did in the run
        self.file = cache_path.open("a")

    def __call__(self, prompt: str) -> str:
        key = hashlib.sha256((self.model + "\0" + prompt).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        err = ""
        for _ in range(4):
            p = subprocess.run(
                ["claude", "-p", "--model", self.model, "--setting-sources", "", "--strict-mcp-config", "--tools", "",
                 "--system-prompt", "You are a careful assistant.", "--no-session-persistence"],
                input=prompt, capture_output=True, text=True, cwd=self.empty, timeout=300,
            )  # fmt: skip
            if p.returncode == 0 and p.stdout.strip():
                with self.lock:
                    if key in self.cache:  # another thread asked the same prompt first: one verdict per prompt
                        return self.cache[key]
                    self.cache[key] = p.stdout.strip()
                    self.file.write(json.dumps({"k": key, "v": self.cache[key]}) + "\n")
                    self.file.flush()
                return self.cache[key]
            err = (p.stderr or p.stdout)[-300:]
        raise RuntimeError(f"claude failed: {err}")


def mcnemar_exact(a: List[bool], b: List[bool]) -> tuple:
    """Two-sided exact McNemar on paired outcomes: (a-only wins, b-only wins, p)."""
    wa = sum(x and not y for x, y in zip(a, b))
    wb = sum(y and not x for x, y in zip(a, b))
    n = wa + wb
    if n == 0:
        return wa, wb, 1.0
    tail = sum(math.comb(n, i) for i in range(min(wa, wb) + 1)) / 2**n
    return wa, wb, min(1.0, 2 * tail)


def run_qa(data, sample, contexts, model, cache_path: Path, arms) -> dict:
    ask = Claude(model, cache_path)

    def one(job):
        (sid, i), arm = job
        qa, conv = data[sid]["qa"][i], data[sid]["conversation"]
        context = "\n".join(contexts[f"{sid}:{i}"][arm])
        answer = ask(READER.format(a=conv["speaker_a"], b=conv["speaker_b"], context=context, question=qa["question"]))
        verdict = ask(JUDGE.format(question=qa["question"], gold_answer=qa["answer"], generated_answer=answer)).upper()
        return (sid, i), arm, "CORRECT" in verdict and "WRONG" not in verdict

    correct: Dict[str, Dict[tuple, bool]] = defaultdict(dict)
    jobs = [(q, a) for q in sample for a in arms]
    with ThreadPoolExecutor(8) as pool:
        for n, (q, arm, ok) in enumerate(pool.map(one, jobs), 1):
            correct[arm][q] = ok
            if n % 200 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    report: dict = {"n": len(sample), "model": model, "arms": {}, "mcnemar": {}}
    # ids and verdicts only, no dataset text
    report["per_question"] = {f"{sid}:{i}": {arm: correct[arm][(sid, i)] for arm in arms} for sid, i in sample}
    for arm in arms:
        words = [sum(len(t.split()) for t in contexts[f"{sid}:{i}"][arm]) for sid, i in sample]
        row = {"words": round(sum(words) / len(words)), "overall": 100 * sum(correct[arm].values()) / len(sample)}
        for cat, name in CATEGORIES.items():
            qs = [q for q in sample if data[q[0]]["qa"][q[1]]["category"] == cat]
            row[name] = round(100 * sum(correct[arm][q] for q in qs) / max(1, len(qs)), 1)
            row[f"n_{name}"] = len(qs)
        report["arms"][arm] = row
    for a, b in (("routed", "turns50"), ("hook_summ", "hook_turns")):
        if a in arms and b in arms:
            wa, wb, p = mcnemar_exact([correct[a][q] for q in sample], [correct[b][q] for q in sample])
            report["mcnemar"][f"{a} vs {b}"] = {f"{a} only": wa, f"{b} only": wb, "p": round(p, 4)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("summaries", help="agent-written summaries, {'<sample_id>|<session>': {'summary': ...}}")
    parser.add_argument("--work", required=True, help="directory for contexts, cache and report (dataset text!)")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--qa", action="store_true", help="also run reader + judge through the Claude CLI")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    assert ADVERSARIAL not in CATEGORIES

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    data = {s["sample_id"]: s for s in json.loads(Path(args.dataset).read_text())}
    sample = sample_questions(list(data.values()), args.n)
    ctx_path = work / "contexts.json"
    if ctx_path.exists():
        contexts = json.loads(ctx_path.read_text())
    else:
        contexts = build_contexts(data, json.loads(Path(args.summaries).read_text()), sample)
        ctx_path.write_text(json.dumps(contexts))
    arms = args.arms.split(",")
    for arm in arms:
        words = [sum(len(t.split()) for t in contexts[f"{sid}:{i}"][arm]) for sid, i in sample]
        print(f"{arm:11} mean words in context {sum(words) / len(words):6.0f}")
    if args.qa:
        report = run_qa(data, sample, contexts, args.model, work / f"cache_{args.model}.jsonl", arms)
        (work / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
