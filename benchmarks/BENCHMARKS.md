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

**LongMemEval-S, session-level Recall@5, raw mode (no LLM): 98.0%** on all
500 questions with `BAAI/bge-small-en-v1.5` (collivind's default embedding
model), standard uncleaned split, nothing excluded (this dataset variant
contains no abstention questions). Measured 2026-07-07; per-question outcomes
in [`results_longmemeval_r5_bge.jsonl`](results_longmemeval_r5_bge.jsonl).

| Question type | bge-small (default) | all-MiniLM-L6-v2 | Questions |
|---|---|---|---|
| knowledge-update | 100.0% | 97.4% | 78 |
| multi-session | 97.7% | 89.5% | 133 |
| single-session-assistant | 100.0% | 100.0% | 56 |
| single-session-preference | 96.7% | 93.3% | 30 |
| single-session-user | 98.6% | 87.1% | 70 |
| temporal-reasoning | 96.2% | 89.5% | 133 |
| **overall** | **98.0%** | **91.8%** | **500** |

Both columns are the same raw pipeline — the only variable is the embedding
model (MiniLM baseline: [`results_longmemeval_r5.jsonl`](results_longmemeval_r5.jsonl),
measured at commit `a02bc89`). We publish the model-selection process too:
`all-mpnet-base-v2` was screened and eliminated (81.5% on a 27-question
subset, at ~15x the compute). No per-question tuning, no heuristics, no
reranking anywhere. A hybrid mode (keyword + temporal boosting in the actual
search engine) is future work and will be reported on the same split with
the same harness.

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
