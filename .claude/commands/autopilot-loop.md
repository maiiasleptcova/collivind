# /autopilot-loop — collivind feature-delivery pipeline

Invoke the `autopilot-loop` skill (`.claude/skills/autopilot-loop/SKILL.md`)
and follow it exactly. Everything below is a quick reference; the skill file
is the source of truth.

- **No arguments** → backlog mode: work through open GitHub issues on
  `maiiasleptcova/collivind`, then eligible ROADMAP.md items, one at a time,
  autonomously, until empty.
- **With arguments** → ad-hoc mode: run the pipeline once for that request.

Pipeline per item: spec → proposer/critic brainstorm (≤2 rounds) → TDD
implementation with a blocking self-review gate (ponytail-review + gates) →
binding VERIFY (pytest, ruff check+format, MCP smoke, benchmark screen when
retrieval is touched, collivind-pro suite when the engine changed) → push as
Maiia + wait for green CI → adversarial review (independent refuter on
approvals) → close the issue (skip on 403) / update ROADMAP → next item.

Orchestrator stays thin: subagents do the work, artefacts live in
`.autopilot/` (gitignored), returns are single JSON lines, never pause
between items. Never start Phase-2/3 architecture bets (team sync server,
knowledge packs, pro licensing) without explicit user approval.
