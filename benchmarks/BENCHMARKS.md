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
model), standard uncleaned split, nothing excluded. That includes the 30
abstention (`_abs`) questions, which LongMemEval's own retrieval eval skips;
without them the figure is 98.3% (462/470). Measured 2026-07-07; per-question outcomes
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

## LoCoMo turn-level retrieval through the full engine

**Task**: [LoCoMo](https://github.com/snap-research/locomo) (`locomo10.json`,
10 conversations, 5,882 turns, 1,986 questions). Unlike the LongMemEval
harness above, which calls the vector store directly, this one goes through the
product path. Every turn is stored with `MemoryManager.add_memory`, which runs
dedup and supersession, enrichment and contradiction detection. Every question
is asked through `MemoryManager.search`, which applies the similarity
threshold, graph expansion and temporal decay. Each run uses a fresh
embedded-mode store in a temp dir, with one project per conversation and the
embed daemon off. The memory text is
`[<session date>] <speaker>: <turn text> [shares a photo: <caption>]`.

**Metrics**: LoCoMo ships gold `evidence` turn ids (`D<session>:<turn>`).
**hit@k** asks whether any evidence turn is among the top-k memories.
**recall@k** is the fraction of the question's evidence turns that are among
the top-k memories. These are retrieval-only metrics: no LLM is involved and no
API key is needed.

**Categories**: LoCoMo stores them as integers. They map as 1 multi-hop,
2 temporal, 3 open-domain, 4 single-hop and 5 adversarial. The README does not
define them; this mapping comes from the eval code, as worked out in
[locomo#29](https://github.com/snap-research/locomo/issues/29). Category 5
questions have no gold answer, only `adversarial_answer`, so they are reported
on their own row and left out of "overall", which matches how Mem0 and Zep
report.

### Results (full run, bge-small-en-v1.5, base commit `eaeacbb`, 2026-11-23)

| Category | Questions | hit@5 | recall@5 | hit@10 | recall@10 |
|---|---|---|---|---|---|
| single-hop | 841 | 61.1% | 59.8% | 71.1% | 69.7% |
| multi-hop | 282 | 59.6% | 32.0% | 71.3% | 43.1% |
| temporal | 321 | 62.3% | 59.5% | 72.3% | 69.6% |
| open-domain | 92 | 40.2% | 28.1% | 48.9% | 38.2% |
| **overall (1–4)** | **1,536** | **59.8%** | **52.7%** | **70.1%** | **62.9%** |
| adversarial (5) | 446 | 37.2% | 36.5% | 51.8% | 51.1% |

- The run took 229 s on an Apple-silicon laptop: 210 s of ingest, with local
  embeddings and no network. Per-question rows are in
  [`results_locomo.jsonl`](results_locomo.jsonl) and the summary is in
  [`results_locomo_summary.json`](results_locomo_summary.json). The rows hold
  ids and scores only: no dataset text, which is CC BY-NC.
- Of the 1,986 questions, 4 have no evidence ids and are skipped, so 1,982 are
  scored. Two more cite a turn id that does not exist; they are scored on the
  ids that do resolve. Evidence ids are normalised before matching: one entry
  lists several ids, and one is zero-padded (`D30:05`).
- Dedup is part of what is measured. The 5,882 turns became 5,539 live
  memories. The other 343 were rejected as exact duplicates or superseded a
  near-duplicate. The harness follows each memory's `superseded_by` chain to
  the live memory and checks that every one of the 5,882 turn ids is still
  reachable from a live memory, so no evidence turn is lost to dedup.
- Nothing benchmark-specific is stored. The harness keeps the
  memory→turn map on its own side, so the embedded text is exactly what that
  turn would produce for a user.
- Every question returned the full 10 results, so the similarity threshold
  never cut a result list short.
- Smoke mode (`--limit 50`, the first 50 questions, all from the first
  conversation) runs in about 20 s. It scores 48 questions, with hit@5 70.8%
  and hit@10 77.1%. It is a regression check only, not a benchmark figure.

### Why this is below the LongMemEval number

The two harnesses differ in granularity and in pipeline: per session through
the raw vector store, against per turn through the full engine. A free check
separates the two. Map each of the top-k LoCoMo turns to its session, and the
same run scores **84.6% at k=5 and 92.4% at k=10** (the session of an evidence
turn is among the sessions of the top-k turns). Most of the gap is therefore
the stricter turn-level target, not the engine. That figure is still not the
same metric as LongMemEval's top-5-sessions R@5. The obvious ways to improve
turn-level hit rate are a hybrid keyword channel, query-time date handling
for temporal questions, and a session-summary memory stored next to the
turns. None of them is in this run.

### Caveats against published LoCoMo numbers

Published LoCoMo figures are **QA accuracy**: Mem0's LLM-judge J score of
66.9 (graph variant 68.4; [paper](https://arxiv.org/abs/2504.19413)), Zep's corrected 75.1
([blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)),
and Letta's filesystem agent at 74.0 ([blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)). Those measure an answering model on top of retrieval, each with its
own prompts, judge and reader model. **Retrieval hit@k is not comparable to
them**, and we do not put the numbers side by side. The dataset also has known
label noise: a [third-party audit](https://dev.to/penfieldlabs/we-audited-locomo-64-of-the-answer-key-is-wrong-and-the-judge-accepts-up-to-63-of-intentionally-33lg)
found about 6% of the answer keys wrong.

### Optional QA mode (not run)

```bash
uv pip install anthropic
ANTHROPIC_API_KEY=... uv run python benchmarks/locomo_bench.py benchmarks/data/locomo10.json --qa [--judge-model claude-haiku-5-5]
```

For each non-adversarial question, QA mode sends the top-k retrieved turns to
a Claude model for a short answer. A second call to the same model then grades
that answer against the gold answer (CORRECT/WRONG), and the result is
reported as `correct`. QA mode is off by default and has **not been run for
this document**. With the default `claude-haiku-5-5` ($0.10/$0.50 per M
tokens), a full run makes about 3,100 calls. At roughly 1.5k input and ≤0.5k
output tokens per call, that is about **$1–2**. With `claude-opus-5-5` the
same run would cost about $40–80. Even then, the prompts and the judge differ
from Mem0's and Zep's, so the earlier caveat still applies.

### Reproducing

```bash
mkdir -p benchmarks/data
curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o benchmarks/data/locomo10.json          # 2.8 MB, CC BY-NC 4.0 (non-commercial)
uv run --extra embedded python benchmarks/locomo_bench.py benchmarks/data/locomo10.json --limit 50   # smoke, ~20 s
uv run --extra embedded python benchmarks/locomo_bench.py benchmarks/data/locomo10.json \
  --summary benchmarks/results_locomo_summary.json                                                   # full, ~4 min
```

`benchmarks/data/` is gitignored. Because of the dataset licences, LoCoMo and
LongMemEval data are never committed.

## Which public memory benchmarks exist (2026-11 survey)

| Benchmark | License | Size | Measures | Gold ids for retrieval | Official scoring |
|---|---|---|---|---|---|
| [LoCoMo](https://github.com/snap-research/locomo) | CC BY-NC 4.0 | 10 convs, ~600 turns each, 1,986 Qs (1,540 non-adversarial) | single-hop, multi-hop, temporal, open-domain, adversarial | yes, turn-level | LLM judge in published work (Mem0 J score) |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | MIT | 500 Qs; S ≈115k tokens / ~40–50 sessions per Q, M ≈1.5M tokens | extraction, multi-session, temporal, knowledge update, abstention | yes, session-level (and `has_answer` turns) | GPT-4o judge |
| [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) | MIT | 146 long-context rows, each with many queries (77 MB) | accurate retrieval, test-time learning, long-range understanding, conflict resolution | not documented | mostly EM / substring EM; LLM judge for some splits |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) | data CC BY-SA 4.0, code MIT | 100 convs, 2,000 Qs, 128K–10M tokens | 10 abilities, including contradiction resolution, event ordering, knowledge update, abstention | the paper mentions source ids; not confirmed in the release | LLM-judged nugget scoring |
| [HaluMem](https://github.com/MemTensor/HaluMem) | CC BY-NC-ND 4.0 | 20 users, 160k–1M tokens each, 3,467 Qs | extraction, update and QA hallucination/omission | yes, memory-point level | LLM judge |

**Why LoCoMo and LongMemEval.** Both are short enough to run locally in
minutes. Both ship gold evidence ids, so they can be scored with no LLM at
all. Between them they cover the abilities a coding-agent memory needs:
single-hop and multi-hop fact recall, temporal questions and knowledge
updates. They are also the two that Mem0, Zep and Letta publish numbers on.

**Why not the others, for now.** MemoryAgentBench feeds book-length contexts
and scores answers, which tests a long-context reader more than a memory
store. BEAM's 1M–10M-token conversations would take hours of local embedding,
and its released files do not clearly include evidence ids. HaluMem's
no-derivatives licence and its focus on LLM extraction quality make it a later
step, once collivind's `extract` path is the thing under test. BEAM's
contradiction-resolution and knowledge-update probes are the natural next
addition, once the contradiction work in progress has landed.

## Search latency at 10k memories

Design target: search under 500 ms at 10k memories. Reproduce with:

```bash
uv run python benchmarks/search_bench.py
```

It seeds 10k memories, each linked to up to 3 entities drawn from a
Zipf(s=1) distribution over 400 entities (the top entity sits on about 15% of
memories, well past the 20-per-entity expansion limit), into the embedded
SQLite stores in a temp dir. Embeddings come from a token-hash embedder, so
**model latency is excluded**: this measures the store and engine only. It
times 200 searches with `limit=10` and graph expansion on, and leaves the
first (vector cache load) out.

| Backend (10k memories) | p50 before | p95 before | p50 after | p95 after |
|---|---|---|---|---|
| embedded (SQLite) | 117 ms | 148 ms | 32 ms | 41 ms |
| docker (Neo4j 5.26 + Qdrant 1.12) | 1804 ms | 2120 ms | 65 ms | 81 ms |

"Before" is origin/main at v0.11.0; "after" is the batched graph expansion.
Both were measured 2026-11-19 on an Apple-silicon laptop. The docker row uses
the same data set, 30 queries per pass, against throwaway containers.
