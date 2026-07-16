---
name: autopilot-loop
description: Full feature-delivery pipeline for collivind — two modes. (A) No arguments: reads the backlog (open GitHub issues on maiiasleptcova/collivind, then ROADMAP.md items), claims the next eligible item, implements it end-to-end, then loops to the next. (B) With arguments (e.g. "/autopilot-loop add LoCoMo benchmark"): implements that specific ad-hoc request once. Both modes use the same pipeline: spec → proposer-critic brainstorm → implementation with self-review gates → verification gate (pytest/ruff/MCP smoke/benchmark) → push + CI → adversarial review. Use whenever the user says "autopilot", "/autopilot-loop", "build this end-to-end", "implement and ship", or "work through the issues".
allowed-tools: Bash, Read, Write, Edit, Task, Glob, Grep
disable-model-invocation: false
---

# Autopilot Loop (collivind)

One command takes a feature request or GitHub issue from raw idea to a verified,
reviewed, pushed change with green CI. It runs a **thin orchestrator + fresh
subagents per role**: each phase gets a clean context window and the main session
never holds implementation noise (specs, diffs, critiques, verification logs).
Every pipeline arrow is an **Agent-tool subagent invocation**; handoffs go through
files in `.autopilot/` and commits in the repo.

## Reference files (load on demand)

- `reference/prompt-templates.md` — full subagent prompt bodies for every step.

## The pipeline

```
[1] Spec: classify area(s) + write spec → .autopilot/specs/<id>.md
→ [2] Brainstorm: R1 proposer+critic (critic may APPROVE → finalise plan = early
  exit; else R2 revised proposer + final critic) → .autopilot/plans/<id>.md
→ [3] Implementer: TDD on a branch or main worktree; self-review gate
  (ponytail-review → correctness → tests) BLOCKING before final commit
→ [4] VERIFY (binding gate): uv run pytest tests/unit tests/mcp -q &&
  uv run ruff check src/ tests/ benchmarks/ && uv run ruff format --check ...
  + MCP smoke (initialize + tools/list over stdio) + benchmark subset when
  search/embedding/storage changed. FAIL → back to [3] (max 2 rework cycles)
→ [5] Push to main as Maiia; watch GitHub Actions until green (CI red → treat
  as VERIFY fail: rework)
→ [6] Review: critic reviews the pushed diff; high-risk diffs get an
  independent second reviewer prompted to REFUTE
→ [7] Close out: comment + close the GitHub issue (skip gracefully on 403 —
  the token may lack Issues permission), update ROADMAP.md if the item lived
  there, record state.
```

## Backlog sources (Mode A)

1. **GitHub issues**: `GET /repos/maiiasleptcova/collivind/issues?state=open`
   (and collivind-pro when the token can read it). Skip issues already
   implemented but unclosable (check git log for "Addresses #N" / "Closes #N").
2. **ROADMAP.md**: "Deferred smaller items" first, then phase items the user
   has green-lit. Never start Phase-2/3 architecture bets (team server, packs,
   pro licensing) without explicit user approval — list them and move on.

No eligible items → print the summary and stop.

## Collivind standards (every feature, every verification)

- **Local-first, no forced LLM keys**: nothing on the default path may require
  a cloud API key. Reject plans that add one.
- **Benchmark honesty**: any change to search, embeddings, enrichment, or
  storage must re-run `benchmarks/longmemeval_bench.py --questions 50` and
  report the number next to the 98.0% baseline; a regression > 1 point on the
  50q screen blocks the change. Never publish tuned-to-the-test heuristics.
- **Embedded mode is single-process**: hooks and CLI must never break session
  start (silent no-op on backend failure); lock contention must stay diagnosed
  (see qdrant_embedded.py).
- **Docs stay true**: README tool tables, BENCHMARKS.md, and ROADMAP.md must
  match the code in the same commit that changes behaviour.
- **Both repos**: after OSS engine changes, run collivind-pro's
  `uv run pytest tests/unit -q` (editable dep on ../collivind-oss).
- **Commit style**: conventional commits, authored as Maiia (local git config).
- **Real exit codes**: never pipe a gate through `| tail` when its exit code
  decides pass/fail — that mask has bitten this repo twice.

## Orchestrator rules (CRITICAL)

1. The main session is the **orchestrator only**: it holds item ID, stage,
   area, file paths, counters — never spec bodies, diffs, critique
   transcripts, or verification logs.
2. Every stage is a fresh `Agent` (Task) subagent: `Plan` type for
   proposer/critic roles, `general-purpose` for spec/implementer/verifier.
3. After a subagent returns, **read only its JSON summary line**. Artefacts
   stay on disk — don't read them back unless routing depends on the content.
4. Prompts stay compact — pass file PATHS + a terse task (full templates in
   `reference/prompt-templates.md`). Returns must be JSON only.
5. **Fully autonomous — do NOT pause between items.** The only stops:
   backlog empty, an environment failure needing the user, a plan that
   requires a Phase-2/3 architecture decision, or user interrupt.

## How to run it

### Step 0 — Dispatch and initialise

Arguments after `/autopilot-loop` → `MODE=adhoc`, `REQUEST=<text>`.
No arguments → `MODE=backlog`: fetch issues (token in the session), pick the
oldest actionable one not yet implemented; else next eligible ROADMAP item.

```bash
SLUG=$(echo "$REQUEST" | head -c 40 | tr ' ' '-' | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-')
FEATURE_ID="feat-$(date +%Y%m%d-%H%M%S)-$SLUG"
mkdir -p .autopilot/{specs,plans,reviews}
SPEC=.autopilot/specs/$FEATURE_ID.md
PLAN=.autopilot/plans/$FEATURE_ID.md
REVIEW=.autopilot/reviews/$FEATURE_ID.md
STATE=.autopilot/state-$FEATURE_ID.json
```

Write `$STATE` (`{"feature_id","request","stage":"initialised","rework_count":0,...}`)
— the source of truth for resume. Work directly on `main` for small items;
use `git worktree add .worktrees/$FEATURE_ID` only when the change is risky
enough to want isolation (`.worktrees/` and `.autopilot/` are gitignored).

### Step 1 — Spec

`general-purpose` agent: read the issue/request + relevant code, classify
area(s) (engine | storage | cli | mcp | hooks | benchmarks | docs), write
`$SPEC` per `.claude/autopilot-templates/spec.md`.
Returns `{"areas":[...],"summary"}`. → `stage: spec_written`.

### Step 2 — Brainstorm (≤2 rounds, early exit on approval)

R1: `Plan`-type proposer writes approach + trade-offs (all serious options
considered, with a recommendation) → `.autopilot/plans/<id>-r1-proposal.md`.
Then a fresh `Plan`-type critic: APPROVE and write final `$PLAN` (per
`autopilot-templates/plan.md`), or write an r1-critique. NEEDS-WORK → R2 with
fresh instances; if R2 still NEEDS-WORK, pause and surface the criticals to
the user. → `stage: plan_ready`.

### Step 3 — Implement

`general-purpose` implementer: reads `$PLAN`, TDD (failing test → code →
green), incremental conventional commits, does NOT push. Before its final
commit it MUST run the self-review gate: `ponytail-review` on its own diff
(fix over-engineering findings), then a correctness pass, then the full local
gates. Returns `{"files_changed","tests_added","commits","summary"}` or
`{"error":"plan-insufficient","detail"}` (→ route back to Step 1 to amend).
→ `stage: implemented`.

### Step 4 — VERIFY (binding)

Orchestrator runs directly (real exit codes, no masking):

```bash
uv run pytest tests/unit tests/mcp -q \
  && uv run ruff check src/ tests/ benchmarks/ \
  && uv run ruff format --check src/ tests/ benchmarks/ \
  && python3 - <<'EOF'   # MCP smoke: initialize + tools/list over stdio
import json, subprocess
p = subprocess.run(["uv","run","python","-m","collivind.mcp.server"],
    input='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n',
    capture_output=True, text=True, timeout=120)
lines=[json.loads(l) for l in p.stdout.strip().splitlines() if l.strip()]
assert any("tools" in (l.get("result") or {}) for l in lines), p.stdout[:500]
print("MCP smoke OK")
EOF
(cd ../collivind-pro && uv run pytest tests/unit -q)   # when engine changed
```

Plus the 50q benchmark screen when search/embedding/storage changed. Any
failure → increment `rework_count`; if < 2, dispatch the implementer again
with the failure log path; else pause and surface. → `stage: verified`.

### Step 5 — Push + CI

`git push origin main` (or push branch + PR for high-risk). Poll
`GET /repos/maiiasleptcova/collivind/actions/runs?per_page=1` until the run
for the pushed SHA completes. Red CI = VERIFY failure → Step 3 rework.
→ `stage: ci_green`.

### Step 6 — Review

Classify the diff first (`git diff --name-only <before>..HEAD`):
- **Trivial** (docs, rename, comment): self-verify only, no agent.
- **Standard**: one `Plan`-type critic reviews the diff → `$REVIEW`; if it
  APPROVEs a non-trivial change, spawn ONE independent `general-purpose`
  reviewer prompted to REFUTE (catches false approvals).
- **High-risk** (engine/dedup/supersede semantics, storage schema, search
  scoring, release/CI workflows): critic + independent refuter always.
Critical findings → Step 3 rework (same pipeline). → `stage: reviewed`.

### Step 7 — Close out

- Issue-sourced: POST a resolution comment + PATCH state closed. On 403,
  note "token lacks Issues permission" in the final output and move on.
- ROADMAP-sourced: update/remove the item in ROADMAP.md (same commit rules).
- Update `$STATE` `stage: done`, print the per-item summary
  (id, commits, verification results, CI run, review verdict, issue status).

### Loop continuation (Mode A)

After close-out, `/compact` preserving mode + completed count + backlog
pointer, then fetch the next item → Step 0. Backlog empty → final summary:
items done / blocked / skipped-needs-user.

## Failure modes

| Symptom | Response |
|---|---|
| GitHub API unreachable | Retry once; then ROADMAP-only mode; note in summary |
| Issues API 403 | Implement anyway; skip comment/close; note in summary |
| VERIFY fails twice | `stage: blocked`, surface logs, next item |
| CI red twice | Same as VERIFY fail |
| Plan needs Phase-2/3 decision | Skip item, list it for the user, next item |
| Embedded store locked during verify | Another collivind process is running — kill stale PIDs per the lock error, retry once |
