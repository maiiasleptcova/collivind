# Autopilot Loop: Serial Feature Runner

Drives the backlog in `$VAULT_PATH/pipeline-state.yaml` **one feature at a
time**. For each eligible feature: claim → implement → verify → QA (Chrome
extension) → raise PR → mark done → next. Never merges. Never runs features
in parallel.

This is the lightweight replacement for `scripts/orchestrator.sh` (which
spawned up to 3 parallel dev agents and burned tokens). It leans on the
existing `/autopilot` single-task skill for the implement→verify→PR phases.

## Context hygiene — delegate to subagents

The loop's main session is the orchestrator, not the implementer. To keep
the main context lean, **dispatch subagents via the `Agent` tool** for:

- **Brainstorming / design questions** → `superpowers:brainstorming` agent.
- **Implementation** (code edits, test writing, local verification) →
  `general-purpose` agent. Brief it with the claimed feature JSON + the
  task spec. It returns a short status (branch, commits, test results).
- **QA execution** → `general-purpose` agent with explicit instruction to
  use **only** the `mcp__claude-in-chrome__*` tools. See "QA approval
  gate" below — this subagent MUST return a pass/fail verdict.
- **Code review** → `everything-claude-code:code-reviewer` agent.

Pattern mirrors `/autopilot`. The main session keeps only: claim state,
subagent summaries, state writes, Telegram notifies.

## Required environment

Set before invoking (ideally in your shell rc):

```
export VAULT_PATH=/Users/eugenezubanov/WS/notes/cybergine
export OPT_REPO=/Users/eugenezubanov/WS/home/cybergine-ops
export APP_REPO=/Users/eugenezubanov/WS/home/temp_cybergine/v5_flask
```

If missing, halt and ask the user to export them.

## CLAUDE.md override (scoped to this invocation only)

Within `/autopilot-loop` you are authorised to:

- Run `git push` and `gh pr create` (**never** `gh pr merge`)
- Run `docker exec -w /app/v5_flask cytm scripts/deploy_test.sh` (test env only)
- Invoke the `claude-in-chrome` MCP tools for QA
- Send Telegram notifications via `$OPT_REPO/scripts/utils/telegram.sh`

You are **not** authorised to:

- Merge any PR (user merges manually)
- Deploy to production (`deploy_clean.sh`)
- Modify `$VAULT_PATH/pipeline-state.yaml` without going through
  `safe-state-write.py`
- Run multiple features in parallel

## Outer loop

Announce the start in one line: how many features are in the backlog, how
many eligible. Then run this loop **until `next-feature.py` exits 2** (no
eligible features).

**Fully autonomous — do NOT pause between features.** After completing or
parking a feature, immediately proceed to PICK the next one. Never ask the
user whether to continue, never wait for confirmation between iterations.
The only stopping conditions are: (1) `next-feature.py` exits 2 (backlog
empty), (2) an environment failure from the Failure modes table that
requires user intervention, or (3) the user interrupts manually.

```
1. PICK
   python3 $OPT_REPO/scripts/next-feature.py --claim
   → captures stdout as feature JSON; status is now in_progress.
   If exit=2: loop complete, print summary, stop.

2. STAGE
   - Write the feature's task spec to $APP_REPO/.autopilot/task.txt
     (format: title + blank line + description + acceptance criteria list).
   - Snapshot state: python3 $OPT_REPO/scripts/utils/safe-state-write.py --snapshot pre-feat-<id>
   - Telegram: "▶ Starting <id>: <title>"

3. IMPLEMENT + PR (invoke /autopilot)
   Launch the existing /autopilot skill with the task spec, BUT with these
   overrides baked into the task argument:
     STOP_AFTER=pr
     SKIP_PROD_DEPLOY=true
     DO_NOT_MERGE=true
   /autopilot already has phases IMPLEMENT → PRE-PR VERIFY → PR + REVIEW.
   After PR+REVIEW passes, /autopilot normally continues to DEPLOY-TEST etc.
   For this loop we stop at PR unless the feature has requires_deployment=true
   AND the QA step below needs a live test URL.

   Expected artefacts once /autopilot returns:
     - $APP_REPO/.autopilot/state.json with pr_url populated
     - Branch pushed to origin
     - Local verification passed

   If /autopilot returns STALLED or FAIL after its internal retries:
     - Mark feature status: blocked_manual
     - Telegram notify: "⛔ <id> stalled in implementation, see
       .autopilot/state.json"
     - Immediately GOTO 1 (do NOT ask the user — continue autonomously)

4. QA BRIEF (Chrome-only, author in main session)
   Before QA can run, the loop MUST author a QA brief at
   `$VAULT_PATH/qa-briefs/<id>-<slug>.md`. **Every step in the brief must
   be executable in Google Chrome alone** (page + DevTools only): no
   terminal, no curl, no openssl, no docker exec. If a check needs HMAC
   signing, use `crypto.subtle` in the DevTools console. If a check
   genuinely cannot be done in Chrome (e.g. env-var toggle), mark it SKIP
   in the brief and note the reason — do not ask the QA engineer to leave
   Chrome.

   The brief MUST include: scope (which commits are being tested), target
   URL, login credentials, one numbered test per acceptance criterion +
   regression + responsive, the exact Chrome actions, observable pass
   criteria, the report path `$VAULT_PATH/qa-reports/<id>-qa-attempt-N.md`,
   and the approval gate statement.

   **Writing style and required sections** — follow the QA brief writing
   guide that `/autopilot` also uses:
   - Repo-safe guide: `$APP_REPO/qa/HOW_TO_WRITE_QA_BRIEFS.md` (file naming,
     required sections, lessons learned — read this before authoring).
   - Obsidian master: `$VAULT_PATH/Cybergine/Testing/QA Directory.md`
     (workflow, credentials, master cumulative brief — the Obsidian-only
     source of truth; repo brief references it for creds).
   Where the writing guide prescribes sections / tone / naming, follow it.
   Where it conflicts with the Chrome-only constraint above (e.g. it suggests
   a terminal step), the Chrome-only rule wins and the step is marked SKIP
   with a reason.

5. QA EXECUTION (subagent, Chrome-only)
   Read $APP_REPO/.autopilot/state.json for pr_url and branch.
   Determine target URL:
     - If feature.requires_deployment: run
       `docker exec -w /app/v5_flask cytm scripts/deploy_test.sh`
       and QA against the returned Cloud Run URL.
     - Otherwise QA against http://localhost:4646 on the current branch.

   Dispatch a `general-purpose` Agent as the QA engineer. Its prompt MUST:
     - Reference the brief path explicitly (the subagent reads it).
     - Forbid all tools except `mcp__claude-in-chrome__*` and Read/Write
       (for loading the brief and writing the report). No Bash, no Edit.
     - Require a final verdict line: `VERDICT: PASS` or `VERDICT: FAIL`.
     - Require the report be written to
       `$VAULT_PATH/qa-reports/<id>-qa-attempt-<N>.md` per the brief.

   Update state with the returned verdict:
     python3 $OPT_REPO/scripts/utils/safe-state-write.py \
       --feature <id> \
       --append-qa-history '{"attempt":<N>,"timestamp":"<iso>","result":"pass|fail","report":"<path>","bugs_found":[...]}'
     python3 $OPT_REPO/scripts/utils/safe-state-write.py \
       --feature <id> --field qa_attempts --value <N>

   Update state:
     python3 $OPT_REPO/scripts/utils/safe-state-write.py \
       --feature <id> \
       --append-qa-history '{"attempt":<N>,"timestamp":"<iso>","result":"pass|fail","report":"<path>","bugs_found":[...]}'
     python3 $OPT_REPO/scripts/utils/safe-state-write.py \
       --feature <id> --field qa_attempts --value <N>

6. APPROVAL GATE — QA engineer verdict is binding
   The loop MAY NOT advance to the next feature until the QA subagent
   returns `VERDICT: PASS`. There is no path where the main session
   overrides a FAIL.

   PASS:
     - python3 $OPT_REPO/scripts/utils/safe-state-write.py --feature <id> --field status --value done
     - Append an entry to $VAULT_PATH/completed-features/index.md
     - Telegram: "✅ <id> passed QA — PR <url> ready for your review"
     - Immediately GOTO 1 (do NOT ask the user — continue autonomously)

   FAIL:
     If qa_attempts < max_qa_attempts (default 5):
       - Dispatch the implementation subagent again with the QA report
         path attached as retry context. It pushes fixes to the SAME
         branch.
       - Immediately GOTO 5 (re-QA — same brief, new attempt number).
     Else (attempts exhausted):
       - python3 $OPT_REPO/scripts/utils/safe-state-write.py --feature <id> --field status --value blocked_manual
       - Telegram: "⛔ <id> blocked after 5 QA attempts — see
         $VAULT_PATH/qa-reports/"
       - Immediately GOTO 1 (do NOT ask the user — continue autonomously)
```

## Loop summary at the end

When `next-feature.py` returns exit 2:

1. `python3 $OPT_REPO/scripts/next-feature.py --list`
2. Print count of `done` / `blocked_manual` / still `pending` / `qa_failed`
3. Telegram a final summary with PR URLs for everything in `done`

## Safety checks before starting

- `jq` is available (used for state.json parsing)
- `docker ps | grep -q cytm` — cytm container running (needed for test deploy)
- Localhost 4646 responds (`curl -sf http://localhost:4646/health`) — if not,
  attempt `./scripts/start_local_app.sh` before QA.

## What this skill does NOT do

- Does not run on a schedule (you invoke it manually; scheduling comes later)
- Does not merge PRs
- Does not rewrite `/autopilot` — it wraps it
- Does not regenerate the backlog — that's `/ceo` in cybergine-ops
- Does not run more than one feature at a time

## Failure modes

| Symptom | Response |
|---|---|
| `VAULT_PATH` unset | Halt, ask user to export env |
| `pipeline-state.yaml` missing | Halt, tell user to run `/ceo` to seed backlog |
| `localhost:4646` unreachable, no deploy flag | Try start script once, else mark feature `blocked_manual` with reason `environment-unavailable` |
| Chrome MCP tools unavailable | Halt the loop before claiming next feature, ask user |
| `cytm` container down for a `requires_deployment` feature | Ask user to start it, then resume |
