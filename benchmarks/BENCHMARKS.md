# Collivind Benchmarks

Every number in this document is reproducible from this repository with the
commands below. Full per-question result files are committed alongside.

## Methodology — what is and isn't measured

**Task**: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (S variant,
500 questions) session-level retrieval. For each question, its haystack
sessions (~50 per question) are ingested into a fresh embedded collivind
store; the question text is the search query; the score is **Recall@5** —
whether an evidence session appears in the top 5 retrieved sessions.

**Mode**: *raw*. Each conversation turn is embedded
(`all-MiniLM-L6-v2`, in-process) and sessions are ranked by their
best-scoring turn. No LLM anywhere in the pipeline, no API keys, no
reranking, no keyword heuristics, no per-question tuning.

**What this does not claim**: R@5 is retrieval recall, not end-to-end QA
accuracy. We deliberately publish no side-by-side comparison against other
memory tools — projects publish different metrics on different splits, and
cross-quoting them is not an honest comparison. Read each project's own
research page for their numbers.

## Results

**LongMemEval-S, session-level Recall@5, raw mode (no LLM): 91.8%** on all
500 questions (this dataset variant contains no abstention questions; nothing
was excluded). Measured 2026-07-07 at commit `a02bc89`; per-question outcomes
in [`results_longmemeval_r5.jsonl`](results_longmemeval_r5.jsonl).

| Question type | R@5 | Questions |
|---|---|---|
| knowledge-update | 97.4% | 78 |
| multi-session | 89.5% | 133 |
| single-session-assistant | 100.0% | 56 |
| single-session-preference | 93.3% | 30 |
| single-session-user | 87.1% | 70 |
| temporal-reasoning | 89.5% | 133 |
| **overall** | **91.8%** | **500** |

The weakest categories (multi-session, temporal-reasoning) are exactly where
collivind's graph expansion and temporal decay should help — the raw mode
above uses neither. A hybrid-mode measurement is future work; it will be
reported on the same split with the same harness.

## Reproducing

```bash
git clone https://github.com/maiiasleptcova/collivind.git
cd collivind
uv sync --extra embedded --dev

# dataset (public, ~278 MB)
mkdir -p benchmarks/data
curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s" \
  -o benchmarks/data/longmemeval_s.json

uv run python benchmarks/longmemeval_bench.py benchmarks/data/longmemeval_s.json
```

Runtime is roughly 8 minutes on an Apple-silicon laptop (local embeddings,
no network). Per-question outcomes land in
`benchmarks/results_longmemeval_r5.jsonl`.
