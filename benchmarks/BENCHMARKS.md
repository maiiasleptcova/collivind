# Collivind Benchmarks

Every number measured here is reproducible from this repository with the
commands below, and names the version, sample size and commit it was measured
on. Result files hold ids and scores only (LoCoMo is CC BY-NC; no dataset text
is committed).

Current release measured: **v0.21.0** (commit `9480b04`), embedded mode, default
config, `BAAI/bge-small-en-v1.5`, 2026-11-26, Apple-silicon laptop.

## Summary

- **LoCoMo, find the exact turn** (1,536 questions, no LLM): collivind returns
  an evidence turn in its top 5 for **72.4%** of questions and in its top 10 for
  **77.5%**. On the same harness, MemPalace's hybrid mode stored per turn gets
  60.5% / 69.5%, plain dense search 56.8% / 66.2%, BM25 57.9% / 65.4%.
- **LoCoMo, find the right session**: collivind 91.9% (top 5) / 96.2% (top 10);
  MemPalace hybrid, its published per-session configuration, 70.4% / 91.9%. A
  random ranking already scores 23.6% / 44.0%, so this metric separates systems
  poorly at top 10.
- **LongMemEval-S, find the right session** (500 questions): every system here
  is between 94.8% and 98.2% at top 5. This benchmark is saturated for
  retrieval and should not be used to rank memory tools.
- **LoCoMo end-to-end answers** (300 questions, Claude Haiku 5.5 as reader and
  judge, scored twice): the prompt hook's default recall answers 70-72%
  correctly; with agent-written summary memories 79-80%; the whole conversation
  in context 76%; the gold evidence alone 84%.
- **Where collivind is behind or unproven**: open-domain questions (44.6%
  strict hit@5); multi-hop answers (47-63% depending on run and summaries);
  search p95 goes from about 40 ms to about 214 ms with the reranker; docker
  mode has no accuracy benchmark; Mem0, Zep, Letta and LangMem were not run
  here (see below).

## Comparison policy

Earlier versions of this file published no side-by-side comparison. That
changed in 0.21.0, with one rule: **only systems run through the same harness,
on the same data, with the same scoring code share a table.** Figures other
projects publish about themselves are listed separately, with their own setup,
and are never ranked against rows measured here. Different reader models,
judges, retrieval depths and splits make cross-quoted numbers incomparable, and
that reason has not gone away.

## Measured here, same harness

Scoring is [`compare.py`](compare.py): one definition per metric, applied to
every system's ids-only ranking file. Per-system scores are in
[`results_comparison.json`](results_comparison.json).

- **turn hit@k**: a gold evidence turn is among the top k units. For collivind
  this is the *strict* variant (the returned memory's text is an evidence
  turn), which is harder than what the other rows get.
- **session hit@k**: a gold evidence session is among the top k sessions,
  sessions ranked by their best unit.
- LoCoMo rows cover categories 1-4 with resolvable evidence (1,536 questions).
  LongMemEval-S rows cover all 500 questions, recall-any.

### LoCoMo, turn level (systems whose stored unit is a turn)

| System | Embedding | turn hit@5 | turn hit@10 |
|---|---|---|---|
| **collivind 0.21.0, default** (hybrid candidates + rerank) | bge-small | **72.4** | **77.5** |
| collivind 0.21.0, `rerank = false` | bge-small | 57.6 | 67.2 |
| MemPalace 3.12.0, hybrid v5, `--granularity dialog` | all-MiniLM-L6-v2 | 60.5 | 69.5 |
| MemPalace 3.12.0, raw, `--granularity dialog` | all-MiniLM-L6-v2 | 49.6 | 58.5 |
| plain dense cosine | bge-small | 56.8 | 66.2 |
| BM25 | none | 57.9 | 65.4 |

Paired exact McNemar tests, collivind default against each other row, give
p < 1e-11 at both depths (for example against MemPalace hybrid at top 5:
collivind alone right on 269 questions, MemPalace alone on 87).

Collivind's numbers are `MemoryManager.search(limit=10)` through the recall
path ([`results_locomo_summary.json`](results_locomo_summary.json)); the
collivind rows in `results_comparison.json` come from the `limit=50` run used
for session ranking, where the rerank pool is larger and turn hit is 72.9 /
78.3. MemPalace's hybrid ranking depends on the requested depth (its rerank
pool is `top_k * 3`); the row above uses a requested depth of 50, the most
favourable to it, and its exact `--top-k 10` run scores 58.6 / 66.1.

### LoCoMo, session level

| System | Stored unit | session hit@5 | session hit@10 |
|---|---|---|---|
| *random ranking (expected)* | | *23.6* | *44.0* |
| **collivind 0.21.0, default** | turn | **91.9** | **96.2** |
| collivind 0.21.0, `rerank = false` | turn | 86.1 | 94.0 |
| BM25 | turn | 89.8 | 95.4 |
| plain dense cosine, bge-small | turn | 85.5 | 93.9 |
| MemPalace hybrid v5, per turn | turn | 85.5 | 93.2 |
| MemPalace raw, per turn | turn | 78.3 | 89.4 |
| MemPalace hybrid v5, per session (published configuration) | session | 70.4 | 91.9 |
| MemPalace raw, per session (published configuration) | session | 51.9 | 65.8 |

Conversations have 19 to 32 sessions, hence the high chance floor. Collivind's
session ranking is derived from `search(limit=50)` (each returned memory's own
session, in order). MemPalace's per-session rows use its own `--top-k 5` and
`--top-k 10` runs; with a requested depth of 50 its hybrid mode scores
82.3 / 92.3. Its published figures use fractional recall over all five
categories including four empty-evidence questions scored as hits: 60.3% raw
and 88.9% hybrid at R@10. Both were reproduced exactly with its unmodified
scripts before anything here was scored.

### LongMemEval-S, session level, all 500 questions

| System | session hit@5 | session hit@10 |
|---|---|---|
| MemPalace hybrid v4 | 98.2 | 99.8 |
| collivind raw vector store (bypasses the engine; 2026-07) | 98.0 | n/a |
| plain dense cosine, bge-small, per turn | 97.8 | 99.0 |
| MemPalace raw | 96.4 | 98.2 |
| BM25, per turn | 94.8 | 97.6 |

Through the full engine, collivind was measured on a 90-question stratified
sample only: 97.8% at top 5, rerank on or off; on those same 90 questions
dense scores 97.8, MemPalace hybrid v4 97.8, MemPalace raw 95.6, BM25 91.1.
With 90 questions one question is 1.1 points. MemPalace's own notes say
hybrid v4 was tuned on three named test questions. Its raw figure here is
96.4% against a published 96.6%: 14 questions in this copy of the dataset carry
one or two extra haystack sessions, and one flips.

### What "MemPalace" means in these rows

Version 3.12.0, commit `87ae502`, run in an isolated environment with no LLM
and none of its CLI installers. Its benchmark scripts do not import the
`mempalace` package: raw and hybrid modes call ChromaDB directly with its
default all-MiniLM-L6-v2 embedder, and the hybrid heuristics (keyword,
quoted-phrase, name and temporal boosts) exist only in those scripts. So these
rows measure MemPalace's published benchmark pipeline, not its shipped product
path, which chunks conversations differently. Modes that need an LLM
(`--llm-rerank`, `diary`, `palace`, `rooms`) were not run. The embedders differ
(bge-small against MiniLM), which is part of each system's default and part of
the gap: collivind's own raw store scored 91.8% on LongMemEval with MiniLM and
98.0% with bge-small.

### LoCoMo end-to-end QA, collivind only

300 questions (categories 1-4, seed 0), every turn stored through
`add_memory`, one reader call per question, reader and judge **Claude Haiku
5.5** through the Claude CLI, judge prompt Mem0's `ACCURACY_PROMPT`.
[`locomo_summary_qa.py`](locomo_summary_qa.py), `--model haiku`. Run 2 repeats
reader and judge with an independent cache on identical contexts, to show
run-to-run variation.

| Context given to the reader | words | run 1 | run 2 | multi-hop (run 1 / run 2) |
|---|---|---|---|---|
| prompt hook, old setting (5 memories, 800 tokens) | 215 | 67.3 | 68.0 | 42.1 / 43.9 |
| prompt hook, shipped default (20 memories, 2,400 tokens) | 764 | 70.3 | 72.0 | 52.6 / 47.4 |
| `search(limit=30)` | 1,015 | 71.3 | | 47.4 |
| shipped hook default, store has summary memories | 1,558 | 79.7 | 78.7 | 63.2 / 57.9 |
| `search(limit=30)`, store has summary memories | 1,864 | 78.7 | | 61.4 |
| whole conversation (no retrieval) | 19,767 | 76.0 | | 66.7 |
| gold evidence turns only (no retrieval) | 73 | 83.7 | | 68.4 |

- The deeper hook default is 3-4 points above the old setting; that is **not**
  significant in either run (paired McNemar p = 0.28 and 0.11).
- Summary memories add 7-9 points and are significant in both runs
  (p = 0.0009 and 0.014). They are written by the agent, one call per session
  at save time; collivind itself runs no model.
- With this reader, collivind with summaries scores above the whole
  conversation in context; the difference is not significant (p = 0.22).
- Earlier QA tables further down used Claude Sonnet 5.5 and are not comparable
  with this one: the reader changes every row.

No other system was run for QA. MemPalace's benchmark scripts are retrieval
only, and the tools that publish QA scores need an LLM at write time.

## Published by others, different setups

Not measured here, not comparable with the tables above or with each other.
Collected from the linked sources on 2026-11-25; check the source for current
figures.

| System | Benchmark and figure | Reader / judge | Retrieval depth | Source |
|---|---|---|---|---|
| Mem0 | LoCoMo LLM-judge 66.9 (graph variant 68.4) | gpt-4o-mini | 30 memories per speaker | [paper](https://arxiv.org/abs/2504.19413) |
| full-context baseline | LoCoMo LLM-judge 72.9 | gpt-4o-mini | whole conversation | same paper |
| Zep | LoCoMo 66.0 in Mem0's paper; 75.1 by Zep's count | see sources | see sources | [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) |
| Letta filesystem agent | LoCoMo 74.0 | see source | agentic file search | [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory) |
| LangMem | LoCoMo LLM-judge 58.1 | gpt-4o-mini | see source | Mem0 paper |
| Mem0 (2026 algorithm) | LoCoMo 92.5, LongMemEval 94.4, vendor claim | see source | see source | [Mem0 blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026) |
| MemPalace | LongMemEval-S R@5 96.6 raw; LoCoMo session R@10 60.3 raw, 88.9 hybrid | none (retrieval) | top 5 / top 10 of 19-32 sessions | [README](https://github.com/MemPalace/mempalace) (reproduced above) |

Mem0, Zep, Letta and LangMem extract or organise memories with an LLM at write
time or run as a hosted service, so they could not be run on this machine,
which has no model API key. Nothing here says how collivind would rank against
them; the only way to know is to run them through the same harness with the
same reader. LoCoMo also has known label noise: a
[third-party audit](https://dev.to/penfieldlabs/we-audited-locomo-64-of-the-answer-key-is-wrong-and-the-judge-accepts-up-to-63-of-intentionally-33lg)
found about 6% of the answer keys wrong.

## Caveats that apply to collivind's own numbers

- The default search path was chosen and measured on the same LoCoMo questions;
  there is no held-out split. The reranker is trained on web search data and
  has been tested on conversational questions only, not on coding prompts.
- Near-duplicate supersession (cosine at or above 0.92) still replaces some
  evidence turns' text with a later near-duplicate; strict scoring counts those
  as misses.
- Retrieval at 0.21.0 is identical to 0.19.0 (same per-question hits), which is
  expected: 0.20 and 0.21 did not change ranking for stores without summaries.
- Search latency was last measured cleanly at 0.19.0 (p50 35 ms / p95 40 ms
  with rerank off, 186 / 214 ms with rerank on, 10k memories, worst-case text
  length). A re-run at 0.21.0 on a heavily loaded machine gave 82 / 158 ms and
  334 / 449 ms and is not a usable measurement.
- Docker mode (Qdrant + Neo4j + HTTP embeddings) has no accuracy benchmark.
- QA scores move 1-2 points between runs with identical contexts.

## Reproducing the comparison

```bash
uv sync --extra embedded --dev
# datasets into benchmarks/data/ (gitignored): locomo10.json, longmemeval_s.json (see below)
uv run --extra embedded python benchmarks/locomo_bench.py benchmarks/data/locomo10.json \
  --summary benchmarks/results_locomo_summary.json                       # collivind, limit 10
uv run --extra embedded python benchmarks/locomo_bench.py benchmarks/data/locomo10.json --top-k 5,10,50 \
  --out benchmarks/data/systems/collivind_locomo_top50.jsonl             # for session ranking
uv run --extra embedded python benchmarks/locomo_bench.py benchmarks/data/locomo10.json --top-k 5,10,50 \
  --search rerank=false --out benchmarks/data/systems/collivind_norerank_locomo_top50.jsonl
uv run --extra embedded python benchmarks/baselines.py locomo benchmarks/data/locomo10.json benchmarks/data/systems
uv run --extra embedded python benchmarks/baselines.py lme benchmarks/data/longmemeval_s.json benchmarks/data/systems
uv run python benchmarks/compare.py benchmarks/data/locomo10.json benchmarks/data/longmemeval_s.json \
  benchmarks/data/systems --out benchmarks/results_comparison.json
```

MemPalace's ranking files (`mempalace_*.json` in the systems directory) come
from its own scripts at commit `87ae502`: `benchmarks/locomo_bench.py` with
`--granularity session|dialog`, `--mode raw|hybrid`, `--top-k 5|10|50`, and
`benchmarks/longmemeval_bench.py` in `raw` and `hybrid_v4` modes, converted to
ids only. Run them with `HOME` redirected to a scratch directory and do not run
its `init` or hook installers.

QA: `uv run --extra embedded python benchmarks/locomo_summary_qa.py
benchmarks/data/locomo10.json SUMMARIES.json --work benchmarks/data/qa --qa
--model haiku --arms hook_old,hook_default,search30,hook_default_summ,search30_summ,full,oracle`
(about 4,200 Claude CLI calls; add `--cache cache_run2.jsonl` for a repeat).
`SUMMARIES.json` holds agent-written session summaries and is not committed.

# Detailed results and history

The sections below are collivind's own measurements as each change landed.
Where they give a figure for an earlier version, the tables above supersede it
for v0.21.0.

## LongMemEval-S through the raw vector store (2026-07)

**Task**: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (S variant,
500 questions) session-level retrieval. For each question, its haystack
sessions (~50 per question) are ingested into a fresh embedded collivind
store; the question text is the search query; the score is **Recall@5** —
whether an evidence session appears in the top 5 retrieved sessions.

**Mode**: *raw*. Each conversation turn is embedded in-process and sessions
are ranked by their best-scoring turn, calling the vector store directly. No
LLM anywhere in the pipeline, no API keys, no reranking, no keyword
heuristics, no per-question tuning. This bypasses `MemoryManager`.

### Results

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
reranking anywhere. This harness measures the raw vector store only; the
engine's hybrid keyword + rerank stage (0.19.0) is measured through
`MemoryManager` in [its own section](#hybrid-candidates-and-cross-encoder-rerank-0190).

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

**strict hit@k** (added in 0.19.0) counts a hit only when a returned
memory's text *is* an evidence turn's text. Plain hit@k credits a memory
with every turn it superseded through near-duplicate dedup, even though the
text returned is the later turn; strict does not.

The current default (0.19.0, hybrid candidates + rerank) is in the
[next section](#hybrid-candidates-and-cross-encoder-rerank-0190); the
committed [`results_locomo.jsonl`](results_locomo.jsonl) and
[`results_locomo_summary.json`](results_locomo_summary.json) are that run.
The table below is the 0.18.0 ranking, which `[search] rerank = false`
still reproduces question for question
([`results_locomo_rerank_off_summary.json`](results_locomo_rerank_off_summary.json)).

### Results before 0.19 (full run, bge-small-en-v1.5, base commit `eaeacbb`, 2026-11-23)

| Category | Questions | hit@5 | recall@5 | hit@10 | recall@10 |
|---|---|---|---|---|---|
| single-hop | 841 | 61.1% | 59.8% | 71.1% | 69.7% |
| multi-hop | 282 | 59.6% | 32.0% | 71.3% | 43.1% |
| temporal | 321 | 62.3% | 59.5% | 72.3% | 69.6% |
| open-domain | 92 | 40.2% | 28.1% | 48.9% | 38.2% |
| **overall (1–4)** | **1,536** | **59.8%** | **52.7%** | **70.1%** | **62.9%** |
| adversarial (5) | 446 | 37.2% | 36.5% | 51.8% | 51.1% |

- The run took 229 s on an Apple-silicon laptop: 210 s of ingest, with local
  embeddings and no network. The rows hold ids and scores only: no dataset
  text, which is CC BY-NC. Strict hit@5 / hit@10 for this ranking are 57.6% /
  67.2%: in 44 of the 1,536 questions the evidence turn's text had been
  replaced by a later near-duplicate (cosine ≥ 0.92).
- Of the 1,986 questions, 4 have no evidence ids and are skipped, so 1,982 are
  scored. Two more cite a turn id that does not exist; they are scored on the
  ids that do resolve. Evidence ids are normalised before matching: one entry
  lists several ids, and one is zero-padded (`D30:05`).
- Dedup is part of what is measured. The 5,882 turns became 5,539 live
  memories. The other 343 were rejected as exact duplicates or superseded a
  near-duplicate. The harness follows each memory's `superseded_by` chain to
  the live memory and checks that every one of the 5,882 turn ids is still
  reachable from a live memory, so no evidence turn id is lost to dedup (its
  text can be: see strict hit@k).
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

## Hybrid candidates and cross-encoder rerank (0.19.0)

**What changed.** Search now has a second stage, all local (no LLM, no API
key). A pool of 50 candidates is fused by reciprocal rank (k = 60) from the
dense top 50 and a keyword (BM25) top 50: SQLite FTS5 in embedded mode, a
Neo4j full-text index in docker mode. The best `limit` graph/entity-only
candidates are added. Each pool candidate is scored against the raw query by
`cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters, 256-token window).
The score is `(sigmoid(logit) × 0.7 + min(graph, 1) × 0.3) × temporal_decay`:
the squashed cross-encoder score takes the cosine's place in the existing
formula, so graph links and recency keep their weight. If no pool candidate
reaches logit 0, the model's own "relevant" boundary, the first-stage
(0.18.0) ranking is served. It is also served when the reranker is missing,
still loading, slow, or returns an error or non-finite scores.
`[search] rerank = false` reproduces 0.18.0 exactly; a golden test pins that,
and on LoCoMo the rerank-off run returns the same ids as 0.18.0 for all
1,982 questions.

### LoCoMo, through `MemoryManager.search`, 1,536 non-adversarial questions

| Arm | strict hit@5 | strict hit@10 | hit@5 | hit@10 | recall@5 | recall@10 |
|---|---|---|---|---|---|---|
| 0.18.0 ranking (`rerank = false`) | 57.6 | 67.2 | 59.8 | 70.1 | 52.7 | 62.9 |
| rerank, dense-only pool (`lexical = false`) | 70.8 | 76.2 | 73.2 | 79.2 | 65.8 | 72.0 |
| **0.19.0 default (hybrid pool + rerank + recall cutoff)** | **72.4** | **77.5** | **74.7** | **80.5** | **67.2** | **73.3** |

| Category (0.19.0 default) | n | strict hit@5 | strict hit@10 | before (strict) |
|---|---|---|---|---|
| single-hop | 841 | 75.9 | 81.1 | 59.2 / 68.6 |
| multi-hop | 282 | 69.1 | 75.2 | 57.8 / 69.1 |
| temporal | 321 | 74.1 | 77.6 | 58.9 / 67.6 |
| open-domain | 92 | 44.6 | 52.2 | 38.0 / 46.7 |
| adversarial (5, not in overall) | 446 | 55.6 | 65.9 | 35.9 / 49.1 |

Run on the final commit, 2026-11-25, Apple-silicon laptop, embed daemon off
(the reranker loads in-process), `HF_HUB_OFFLINE=1`, questions asked through
the recall path (`min_rerank_score` = the default `recall_min_score`): 303 s
against 168 s for rerank off. Reproduce with
`uv run --extra embedded python benchmarks/locomo_bench.py benchmarks/data/locomo10.json`
and `--search rerank=false` / `--search lexical=false` for the other arms.

An offline estimate over raw turns (reranking an RRF(dense, BM25) top 50)
was 76.4 / 82.2. The gap has known parts. In 36 (hit@5) and 45 (hit@10)
questions the evidence turn's text was replaced by a near-duplicate at
ingest; those count as hits by id but not by text (2.3 and 2.9 points). The
relevance gate costs another 0.9 / 0.6 points here (below); the recall
cutoff costs nothing measurable.

**How the gate was chosen.** With the plain formula, LongMemEval turn hit@10
fell (83.3 → 80.0, 5 losses against 2 wins). In the losses the cross-encoder
scored every candidate near 0 (sigmoid ≤ 0.003). The questions were indirect
("any tips on what to look for in a new guitar?" where the answer turn is the
user's earlier mention of their guitars), and the model's order among texts
it found irrelevant threw away a dense ranking that had the answer at rank
0–8. Coding prompts are often indirect in the same way. Two fixes were
declared before looking at any pool data: the gate above, and a blend of
cosine back into the score at a fixed weight of 0.5. Both were scored offline
on logged pools, with no sweep. Strict hit@5 / hit@10:

| Formula | LoCoMo (1,536) | LongMemEval seed 0 (90) | LongMemEval seed 1 (90, 69 unseen) |
|---|---|---|---|
| first stage (0.18.0) | 57.6 / 67.2 | 74.4 / 83.3 | 82.2 / 91.1 |
| rerank, plain formula | 73.2 / 78.1 | 74.4 / 80.0 | 85.6 / 91.1 |
| rerank + gate (shipped) | 72.3 / 77.5 | 75.6 / 83.3 | 86.7 / 93.3 |
| rerank + cosine blend | 71.7 / 78.4 | 74.4 / 83.3 | 86.7 / 92.2 |

The gate was kept because it is best at k = 5 (what the prompt hook injects)
on all three sets, and because it reuses the fallback path that is already
tested. LongMemEval "seed 1" is a second stratified sample (seed 1); 21 of
its 90 questions overlap seed 0.

### LongMemEval-S through `MemoryManager`, turn level

Session R@5 is saturated (about 98%) and cannot show a ranking change.
[`longmemeval_engine.py`](longmemeval_engine.py) adds **turn_hit@k**: one of
the dataset's `has_answer` turns, by text, is among the top-k memories.
Every turn is stored with `add_memory`; the sample is 15 questions per type
(seed 0, abstention excluded, 90 questions), the same sample as the
2026-11-25 evaluation. Both arms search the same store through the recall
path (the default cutoff is on; it is inert with rerank off).

| Arm (90 questions) | session R@5 | turn hit@5 | turn hit@10 |
|---|---|---|---|
| rerank off (0.18.0 ranking) | 97.8 | 74.4 | 83.3 |
| **0.19.0 default** | **97.8** | **75.6** | **83.3** |

| Type (15 each) | session R@5 on / off | turn hit@5 on / off | turn hit@10 on / off |
|---|---|---|---|
| knowledge-update | 100.0 / 100.0 | 86.7 / 86.7 | 93.3 / 100.0 |
| multi-session | 100.0 / 100.0 | 93.3 / 86.7 | 100.0 / 100.0 |
| single-session-assistant | 100.0 / 100.0 | 66.7 / 60.0 | 73.3 / 66.7 |
| single-session-preference | 100.0 / 100.0 | 80.0 / 80.0 | 93.3 / 93.3 |
| single-session-user | 100.0 / 100.0 | 66.7 / 66.7 | 66.7 / 66.7 |
| temporal-reasoning | 86.7 / 86.7 | 60.0 / 66.7 | 73.3 / 73.3 |

621 s for both arms. Reproduce with
`uv run --extra embedded python benchmarks/longmemeval_engine.py benchmarks/data/longmemeval_s.json --per-type 15`.

Rows (ids and scores only):
[`results_longmemeval_engine.jsonl`](results_longmemeval_engine.jsonl),
summary [`results_longmemeval_engine_summary.json`](results_longmemeval_engine_summary.json).
With 90 questions one question is 1.1 points.

### Recall relevance cutoff and hook depth (0.19.0)

The prompt hook now recalls up to `[hooks] recall_limit = 20` memories, up
from a hard-coded 5. Mem0's own LoCoMo eval retrieves 30 per speaker, and
the QA sample below scores 60.7% at 5, 66.0% at 10 and 76.3% at 30. At 20,
a prompt unrelated to everything stored would fill all 20 slots, so recall
gets a cutoff. `[search] recall_min_score = 0.00002` drops a memory when
sigmoid(cross-encoder logit) is below it. It applies after ordering (gated
or reranked), so recall can return fewer than `limit` memories, or none;
the hook then prints nothing. It applies only on the recall path
(`get_context`: the prompt hook and `collivind_get_context`), only when the
reranker actually ran, and never to explicit search. The LoCoMo numbers in
this section are measured with the cutoff on, because the harness asks
through the recall path.

**Choosing the default.** Cross-encoder scores were logged for every
returned candidate through the real engine (limit 10). LoCoMo barely
separates the two cases. Evidence turns score at the median 0.977
(p10 0.087), and other turns score at the median 0.294 (p10 0.0018),
because they come from the same conversation. Adversarial (category 5)
questions are about the conversation too: their best-candidate median is
0.953. So LoCoMo can only bound the cost:

| cutoff | strict hit@5 | strict hit@10 | kept of 10 (cat 1–4) | no result (cat 1–4) | kept of 10 (cat 5) |
|---|---|---|---|---|---|
| 0 (off) | 72.3 | 77.5 | 10.0 | 0.0% | 10.0 |
| **0.00002** | **72.4** | **77.5** | **10.0** | **0.0%** | **10.0** |
| 0.0001 | 72.5 | 77.3 | 9.9 | 0.0% | 9.9 |
| 0.0003 | 72.5 | 77.4 | 9.7 | 0.1% | 9.6 |
| 0.001 | 72.3 | 77.0 | 9.5 | 0.2% | 9.0 |
| 0.003 | 71.7 | 76.2 | 9.0 | 0.7% | 8.1 |
| 0.01 | 71.2 | 75.0 | 8.4 | 1.2% | 7.0 |
| 0.1 | 69.0 | 71.5 | 6.5 | 3.5% | 4.7 |
| 0.5 | 63.5 | 65.2 | 4.4 | 9.4% | 2.8 |

Up to 0.001 the cost stays within 1 point of strict hit@10. LoCoMo is
conversational and the reranker is MS MARCO-trained, so the cutoff was also
checked against a synthetic store of 20 coding memories (decisions, errors,
snippets across several projects) with 10 coding prompts that each target
one memory and 10 prompts unrelated to all of them, at limit 20:

| cutoff | relevant: target kept | relevant: no result | unrelated: mean kept | unrelated: no result |
|---|---|---|---|---|
| 0 (first stage, or rerank without cutoff) | 10 / 10 | 0 | 20 | 0 / 10 |
| 0.00001 | 10 / 10 | 0 | 20 | 0 / 10 |
| **0.00002** | **10 / 10** | **0** | **0.2** | **8 / 10** |
| 0.0001 | 10 / 10 | 0 | 0.1 | 9 / 10 |
| 0.0003 | 9 / 10 | 1 | 0.1 | 9 / 10 |
| 0.001 | 8 / 10 | 2 | 0.0 | 10 / 10 |

Relevant coding targets scored from 0.00016 to 0.9998, and the indirect
phrasings sat lowest: "how do we decide if two products are duplicates
after a rebuild" → 0.00016, "how do I wipe and recreate my local database"
→ 0.00044. Unrelated prompts score at the model's floor, 1.1e-5 to 1.4e-5
(sigmoid of a logit near −11.3). Two were exceptions: "what's the time
complexity of quicksort?" scored 2.3e-5 and "convert this CSV to JSON"
scored 8.4e-4.

LongMemEval decides between "just above the floor" and anything higher.
The first default tried, 0.0001, passed LoCoMo (72.5 / 77.3) but cost
LongMemEval session R@5 97.8 → 94.4 and turn hit@10 83.3 → 80.0 through the
real engine. In those questions ("any tips on what to look for in a new
guitar?") the answer turn itself scores below 0.0001: the same indirect
matches the relevance gate keeps for. Counting questions whose best answer
turn falls below the cutoff, on logged pools:

| cutoff | LME seed 0 (90) | LME seed 1 (90) |
|---|---|---|
| 0.00001 | 0 | 0 |
| 0.000015 | 2 | 0 |
| **0.00002** | **2** | **0** |
| 0.00003 | 4 | 1 |
| 0.0001 | 7 | 5 |

Two seed-0 answer turns score at the floor itself, the same as text
unrelated to the query. No cutoff can keep them and still empty unrelated
prompts. **0.00002 was taken**: just above the floor. It costs nothing
measurable on LoCoMo and keeps every coding target. It empties 8 of 10
unrelated coding prompts (the other two recall one memory each instead of
20). On LongMemEval it changes nothing measurable: the two floor-level
answer turns were not in the top 10 anyway, and the turn and session
numbers below are the same with and without it. Raise it to cut more noise at the price of indirect matches, or set
0 to disable. **The coding check is small (20 memories, 20 prompts written
for it) and is a sanity check, not a benchmark**; the score scale of a
different `rerank_model` needs its own cutoff.

**Injected tokens at hook depth and budget.** These come from the real
`get_context`, with tokens counted as get_context's budget estimates them
(len/4). The hook now defaults to `[hooks] recall_limit = 20` and
`recall_max_tokens = 2400`; it was 5 memories and 800 tokens before.
`hook user-prompt --max-tokens` still overrides the budget. The recall
cutoff runs before the budget is packed. Two stores were measured:
- LoCoMo: 150 questions over two conversations; memories have a median of
  164 characters, about 41 tokens.
- A synthetic coding-length store: 48 memories, each built from four joined
  coding facts, median 408 characters, about 102 tokens. It was queried with
  the 10 relevant and 10 unrelated coding prompts from the cutoff check.

| Store / prompts | limit, budget | memories injected | tokens (mean / max) | nothing recalled |
|---|---|---|---|---|
| LoCoMo | 5, 800 (old default) | 5.0 | 316 / 433 | 0 / 150 |
| LoCoMo | 10, 800 | 9.9 | 592 / 798 | 0 / 150 |
| LoCoMo | 20, 800 | 13.7 | 773 / 811 | 0 / 150 |
| LoCoMo | **20, 2400 (new default)** | **19.8** | **1,098 / 1,548** | 0 / 150 |
| coding, relevant | 5, 800 (old default) | 5.0 | 551 / 601 | 0 / 10 |
| coding, relevant | 20, 800 | 6.7 | 737 / 804 | 0 / 10 |
| coding, relevant | **20, 2400 (new default)** | **11.9** | **1,313 / 2,241** | 0 / 10 |
| coding, unrelated | 5, 800 (old default) | 0.8 | 83 / 507 | 8 / 10 |
| coding, unrelated | **20, 2400 (new default)** | **1.4** | **146 / 1,024** | **8 / 10** |

At 800 tokens, depth 20 was capped at about 6.7 coding-length memories. At
2,400 a relevant coding prompt now costs about 1,300 tokens on average
(about 12 memories; the cutoff, not the budget, trims the rest) and at most
about 2,240. An unrelated prompt usually costs nothing: 8 of 10 recalled
nothing, and the other two recalled a few memories.

### End-to-end QA on LoCoMo (300-question sample, from the 2026-11-25 evaluation)

**Not comparable to vendor-published LoCoMo scores.** This is a 300-question
sample (seed 0, categories 1–4). The reader and the judge are both Claude
Sonnet 5.5; Mem0's published runs use gpt-4o-mini. The judge prompt is
Mem0's `ACCURACY_PROMPT` verbatim, and there is one reader call per question.
The binomial 95% CI is about ±5 points. "Hybrid + rerank" here is the
offline estimate over raw turns (RRF(dense, BM25) top 50 reranked by the
same cross-encoder, no gate), not 0.19.0's engine path.

| Context given to the reader | overall | single-hop | multi-hop | temporal | open-domain |
|---|---|---|---|---|---|
| n | 300 | 167 | 57 | 54 | 22 |
| collivind top 5 (what the UserPromptSubmit hook injects) | 60.7 | 72.5 | 33.3 | 57.4 | 50.0 |
| hybrid + rerank, top 5 | 71.7 | 82.6 | 38.6 | 74.1 | 68.2 |
| collivind top 10 (search default) | 66.0 | 76.6 | 38.6 | 68.5 | 50.0 |
| hybrid + rerank, top 10 | 73.3 | 83.2 | 47.4 | 74.1 | 63.6 |
| collivind top 30 | 76.3 | 84.4 | 52.6 | 77.8 | 72.7 |
| hybrid + rerank, top 30 | 78.0 | 89.2 | 49.1 | 77.8 | 68.2 |
| collivind top 50 | 77.3 | 84.4 | 59.6 | 81.5 | 59.1 |
| hybrid + rerank, top 50 | 77.0 | 86.8 | 52.6 | 79.6 | 59.1 |
| full conversation in context | 86.0 | 92.8 | 77.2 | 83.3 | 63.6 |
| oracle (gold evidence turns only) | 88.7 | 94.6 | 68.4 | 90.7 | 90.9 |

Paired exact McNemar tests on the same questions: at top 5, rerank beats
the engine (45 wins to 12, p = 0.00001); at top 10 too (p = 0.005). At top
30 (p = 0.49) and top 50 (p = 0.88) there is no difference. So rerank
matters at small k, and context depth is a separate lever.

### Docker mode

The keyword channel is a Neo4j full-text index on `Memory.content`
(`memory_content_fulltext`). `initialize()` creates it, and a search on an
older store that finds it missing creates it too; Neo4j fills it from the
existing memories in the background. The reranker is `POST /rerank` on the
embeddings container. An image built before 0.19 has no such endpoint and
search keeps the first-stage ranking until `collivind docker up` rebuilds
it. Checked end to end on throwaway containers (Neo4j 5.26, Qdrant 1.19.2,
the rebuilt embeddings image, 768 MB limit, about 455 MB resident with both
models): the keyword-only hit comes back reranked, 57–72 ms per search
warm, `/rerank` of 50 texts about 64 ms. The LoCoMo and LongMemEval numbers
above are embedded mode only; docker-mode accuracy was not benchmarked, and
neither was the cost of the extra `get_memory` round trips (up to about 50)
per search on Neo4j at scale.

## Summary memories (0.20.0)

A summary memory is an agent-written list of dated facts over one session's
memories (or a named topic), linked to each member by `SUMMARIZES`. At recall
the best summaries for the query are ranked against each other by their best
fact line; those the reranker judges relevant (best-line logit at least 0)
come first, then their most relevant members merged by score with the best
other memories (see README, *How It Works*). This differs from the offline
"routed" arm in the 2026-11-25 evaluation in three ways: best-line scoring
and 20 dense summary candidates (chosen from the retrieval check below), and
the relevance gate and score ordering (added after code review so an
off-topic summary cannot lead a recall). Collivind never writes summary text: **every summary costs one agent
call per session at save time** (the agent reads the session's memories with
`collivind_summarize` and writes the facts). That cost is not in the numbers
below.

### LoCoMo QA through `MemoryManager` (300-question sample, 2026-11-25)

Same sample, reader, judge and prompts as the end-to-end QA above (seed 0,
categories 1-4, Claude Sonnet 5.5 reader and judge through the Claude CLI,
Mem0's judge prompt). Every turn is added with `add_memory` (one session id
per LoCoMo session); the summaries are the 272 written by Sonnet 5.5 from
the session text alone in the 2026-11-25 evaluation (mean 327 words, about
45% of a session), stored with `save_summary`. Both arms run on the same code
and store: the turns-only searches run before any summary is saved.
`benchmarks/locomo_summary_qa.py` reproduces it; results (ids and verdicts
only) in `results_locomo_summary_qa.json`.

| Context given to the reader | words | overall | single-hop | multi-hop | temporal | open-domain |
|---|---|---|---|---|---|---|
| n | | 300 | 167 | 57 | 54 | 22 |
| turns only, `search(limit=50)` | 1,648 | 75.7 | 85.6 | 56.1 | 72.2 | 59.1 |
| with summaries, `search(limit=28)`: 3 summaries + their best members + other turns | 1,803 | **83.7** | 93.4 | 57.9 | 85.2 | 72.7 |
| prompt hook at 800 tokens (`get_context`, limit 20; the old default), turns only | 560 | 74.3 | 84.4 | 49.1 | 74.1 | 63.6 |
| prompt hook at 800 tokens, with summaries | 569 | **81.3** | 93.4 | 47.4 | 85.2 | 68.2 |

Paired exact McNemar: with summaries vs turns only, 32 questions to 8,
p = 0.0002; at the hook's 800-token budget, 38 to 17, p = 0.007. These arms
were measured before the hook default rose to 2,400 tokens
(`[hooks] recall_max_tokens`) and were not re-run at 2,400. The routed context's
gold-session summary is among the 3 returned for 270 of 300 questions.

**Multi-hop barely moved here** (57.9 vs 56.1), unlike the offline estimate
in the evaluation (64.9 vs 52.6), and is below the 62% this work aimed for.
The contexts are not the reason: for the 57 multi-hop questions the share of
gold sessions among the routed summaries is the same as offline, and paired
against the offline routed arm the engine's answers differ on 3 vs 7
questions (p = 0.34). With 57 questions one question is 1.8 points and the
95% CI is about ±13, so the multi-hop column cannot separate these arms.
Not tuned further.

**What was changed to get here, and why (measured without the reader).**
The engine's cross-encoder reads 256 tokens, so a 330-word summary scored
whole is judged on its first facts: the gold session's summary reached the
top 3 for 207 of 300 questions (259 at 512 tokens). Scoring each summary by
its best line gives 277 at 256 tokens, without changing how ordinary
memories are scored. The 20 dense summary candidates come from the same
check (top 10: 245, top 20: 271, all: 277); a LoCoMo conversation has at
most 35 summaries, so on a store with hundreds of summaries the dense
prefilter matters more than here. Line scoring is capped at 300
cross-encoder pairs per search.

**Caveats.** One reader run; the binomial 95% CI is about ±5 points overall
and ±13 on multi-hop. Verdicts are cached by prompt, one per prompt. The
summaries came from Sonnet 5.5 with a fixed prompt; an agent summarising
coding sessions mid-task will write different ones. Measured on base `6651c12`; the later base commit
changes a test fixture only. Embedded mode only; routing on Neo4j costs one
round trip per member read (up to 120 per search). Routing adds about 150 ms
median per prompt recall on a laptop (127 to 279 ms, 60 summaries in the
project); ten unrelated prompts on the same store recalled nothing, with or
without summaries.

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

**With rerank (0.19.0).** The same script now runs a second arm with the
real cross-encoder loaded in-process over the real pool (dense + keyword
top 50, plus graph-only candidates and the first-stage top `limit`: 60
texts on average at limit 10). Each text is padded
to fill the model's 256-token window, so this is the worst case: real
memories are shorter. Embedded SQLite, 10k memories, 200 queries,
2026-11-25, Apple silicon:

| Arm | p50 | p95 | max |
|---|---|---|---|
| rerank off | 34.9 ms | 40.4 ms | 57.3 ms |
| rerank on (worst-case text length) | 185.6 ms | 214.1 ms | 242.2 ms |
| limit 20 (the hook's new depth), rerank off | 45.6 ms | 58.3 ms | 107.6 ms |
| limit 20, rerank on (pool 70) | 219.2 ms | 281.2 ms | 317.5 ms |

Through the resident embed daemon (what hooks use), a 50-text rerank of
LoCoMo-length turns took about 25 ms warm. The first request after the
daemon starts took about 1 s.
Both were measured 2026-11-19 on an Apple-silicon laptop. The docker row uses
the same data set, 30 queries per pass, against throwaway containers.
