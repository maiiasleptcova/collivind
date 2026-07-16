# {{Feature title — derived from the request}}

**Feature ID:** {{feat-YYYYMMDD-HHMMSS-slug}}
**Area(s):** {{engine | storage | cli | mcp | hooks | benchmarks | docs}}
**Source:** {{github issue #N | ROADMAP.md item | ad-hoc request}}
**Created:** {{ISO timestamp}}

---

## Problem statement

What's broken, missing, or suboptimal right now. Describe the PROBLEM, not the
solution. Two to four sentences max.

## User story

As a {{developer using collivind with Claude Code/Codex | team sharing memory |
collivind operator}}, I want {{outcome}} so that {{underlying goal}}.

## Context

Background the implementer needs — prior decisions (check ROADMAP.md and recent
commits), related shipped features, constraints. Collivind invariants that apply:
local-first / no forced LLM keys, embedded mode is single-process, LongMemEval
R@5 98.0% baseline, MIT open core with collivind-pro depending on this engine.

## Acceptance criteria

Numbered. Testable. Each verifiable via pytest, a CLI invocation, or an MCP
JSON-RPC call — no browser, no cloud account.

1. {{Specific behaviour under specific conditions}}
2. ...

## Out of scope

- {{Thing deliberately not being done — e.g. "Phase-2 sync server"}}

## Open questions

Only blockers the spec author genuinely couldn't resolve from the repo.

- {{Question for the user, or "none"}}
