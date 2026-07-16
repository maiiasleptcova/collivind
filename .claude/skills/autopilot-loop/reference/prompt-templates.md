# Autopilot subagent prompt templates (collivind)

Compact bodies for each pipeline stage. Substitute `{...}` placeholders.
Every prompt ends with the same contract: *"Your final message must be a
single JSON line with exactly the keys listed — no prose around it."*

## Step 1 — Spec writer (`general-purpose`)

```
Write an implementation spec for this collivind item.

Request/issue: {REQUEST_TEXT_OR_ISSUE_PATH}
Repo: /Users/eugenezubanov/WS/home/collivind-pro-and-oss/collivind-oss

1. Read the request and the code it touches (start from src/collivind/,
   README.md, ROADMAP.md). Collivind is a local-first graph+vector memory
   MCP server + CLI; do not propose cloud-key-dependent defaults.
2. Classify the affected area(s): engine | storage | cli | mcp | hooks |
   benchmarks | docs.
3. Write the spec to {SPEC} following .claude/autopilot-templates/spec.md:
   problem, user story, context, numbered TESTABLE acceptance criteria
   (verifiable via pytest, CLI invocation, or MCP JSON-RPC — no browser),
   out of scope, open questions.

Return JSON: {"areas": [...], "summary": "<one sentence>", "open_questions": n}
```

## Step 2a — Proposer (`Plan` agent)

```
Design the implementation approach for a collivind feature.

Spec: {SPEC}  (read it; read any code it references)

Consider ALL serious options (including "do less"), with trade-offs against
collivind's constraints: local-first, no forced LLM keys, embedded mode is
single-process, benchmark honesty (LongMemEval R@5 98.0% baseline must not
regress), MIT open core with pro repo depending on this engine.

Write your proposal to {PROPOSAL}: options considered, recommendation,
rejected alternatives with one-line reasons, risk list, test strategy.

Return JSON: {"recommendation": "<one sentence>", "options_considered": n}
```

## Step 2b — Critic (`Plan` agent, fresh instance)

```
You are the critic. Read spec {SPEC} and proposal {PROPOSAL}.

Attack it: correctness holes, simpler alternatives it ignored (YAGNI),
hidden coupling with dedup/supersede/enrichment/search scoring, benchmark
or docs impact it missed, test gaps. Judge against collivind's standards
in .claude/skills/autopilot-loop/SKILL.md § Collivind standards.

If the approach is sound: write the FINAL plan to {PLAN} following
.claude/autopilot-templates/plan.md and verdict APPROVED.
If not: write your critique to {CRITIQUE} and verdict NEEDS-WORK.

Return JSON: {"verdict": "APPROVED"|"NEEDS-WORK", "summary": "<one sentence>"}
```

## Step 3 — Implementer (`general-purpose`)

```
Implement the plan at {PLAN} in
/Users/eugenezubanov/WS/home/collivind-pro-and-oss/collivind-oss.

Rules:
- TDD: failing test first, then the code, then green. pytest + mocks for
  unit tests (see tests/unit/ conventions).
- Match repo style: dataclasses not pydantic, stdlib-first, ruff line 120.
- Conventional commits, incremental. Do NOT push.
- {RETRY_CONTEXT: on rework, path to the failure log / review findings}

Self-review gate before your FINAL commit (all blocking):
1. Run the ponytail-review skill on your own diff; fix over-engineering
   findings (reinvented stdlib, speculative abstraction, dead flexibility).
2. Correctness pass: reread the diff hunting for the bug you'd be paged for.
3. Gates: uv run pytest tests/unit tests/mcp -q && uv run ruff check src/
   tests/ benchmarks/ && uv run ruff format --check src/ tests/ benchmarks/
   (run ruff format first if needed). Real exit codes — no piping to tail.
4. Docs in the same commit when behaviour changed (README/BENCHMARKS/ROADMAP).

Return JSON: {"files_changed": n, "tests_added": n, "commits": [...],
"summary": "<one sentence>"}  — or {"error": "plan-insufficient",
"detail": "<what's missing>"}
```

## Step 6 — Reviewer critic (`Plan` agent)

```
Review the pushed collivind diff {DIFF_RANGE} (git show/diff it yourself) in
/Users/eugenezubanov/WS/home/collivind-pro-and-oss/collivind-oss.

Hunt: correctness bugs, dedup/supersede/search-scoring semantic breaks,
embedded-mode multi-process hazards, silent-failure paths in hooks (they
must never break session start), benchmark validity, docs drift, missing
tests. Write findings to {REVIEW} ranked by severity.

Return JSON: {"verdict": "APPROVE"|"FINDINGS", "critical": n, "major": n,
"summary": "<one sentence>"}
```

## Step 6b — Independent refuter (`general-purpose`)

```
An implementation was just approved. Your job is to REFUTE that approval.
Diff: {DIFF_RANGE} in /Users/eugenezubanov/WS/home/collivind-pro-and-oss/collivind-oss.
Assume the approver missed something: trace every changed code path for the
input/state that breaks it, check the tests actually assert the new
behaviour (not just run it), check concurrency with a second collivind
process. If you genuinely cannot break it, say so.

Return JSON: {"refuted": true|false, "findings": [...], "summary": "<one sentence>"}
```
