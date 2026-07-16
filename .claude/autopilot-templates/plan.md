# Implementation Plan — {{Feature title}}

**Feature ID:** {{feat-id}}
**Area(s):** {{areas}}
**Critique outcome:** {{APPROVED-R1 | APPROVED-R2}}
**Based on proposal:** {{path to final proposal file}}

---

## Approach summary

Two to four sentences: the shape of the solution, the key decisions, why this
over the rejected alternatives.

## Architecture decisions

The 2–4 choices that would force a re-plan if changed later, each with a
one-line justification.

- **{{Decision}}** — {{why}}

## Changes by file

| File | Change |
|---|---|
| `src/collivind/...` | {{what and why}} |
| `tests/unit/test_...` | {{new/changed tests}} |
| `README.md` / `ROADMAP.md` / `BENCHMARKS.md` | {{doc sync, or "n/a"}} |

## Interfaces

New/changed function signatures, CLI flags, MCP tool schemas, config fields
(remember: config template + dataclass + load_config all move together — the
drift-guard test enforces [search]/[hooks]).

## Test strategy

- Unit: {{what gets mocked, what gets asserted}}
- MCP smoke / CLI invocation: {{if applicable}}
- Benchmark screen (50q) required: {{yes — search/embedding/storage touched | no}}
- collivind-pro suite required: {{yes — engine surface changed | no}}

## Risks and edge cases

- {{Risk}} → {{mitigation or accepted-with-reason}}

## Rollback

{{Usually: revert the commit. Note anything stateful — schema, config default,
vector-store format — that makes rollback non-trivial.}}
