"""Automatic capture state: per-session checkpoints, confirmations, heartbeats (#35).

A hook can only *ask* the agent to save; the write happens later, through an
MCP tool call the hook never sees. So a reminder is not a save. Each reminder
carries a checkpoint id, the agent passes it back to `collivind_batch_add`, and
the MCP server records the result here. Until then the checkpoint stays
pending and is re-requested, so "reminded" can never be mistaken for "saved".

Everything lives in two small files in the data dir:
- hook_state.json — per-session counters, pending checkpoints, hook heartbeats
- checkpoints.jsonl — append-only history of confirmed saves and failures
"""

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Iterator, Optional

fcntl: Optional[ModuleType]
try:
    import fcntl as _fcntl

    fcntl = _fcntl
except ImportError:  # Windows: no flock; state updates are then unlocked
    fcntl = None

# Re-ask after this many further responses if a requested save never landed,
# doubling the wait each time; after MAX_RETRIES re-asks (and one more wait)
# the checkpoint is recorded as failed instead of nagging for the session TTL.
RETRY_AFTER = 3
MAX_RETRIES = 3
# Keep the state file bounded: drop sessions idle longer than this.
SESSION_TTL_DAYS = 14


def _dir() -> Path:
    return Path.home() / ".collivind"


def state_path() -> Path:
    return _dir() / "hook_state.json"


def history_path() -> Path:
    return _dir() / "checkpoints.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def locked_state() -> Iterator[dict]:
    """Read-modify-write hook_state.json under an exclusive lock.

    Concurrent sessions run hooks at the same time; without the lock one
    session's write silently drops another's counter (#35: "one session must
    not consume or reset another's save interval").
    """
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        state.setdefault("sessions", {})
        state.setdefault("heartbeats", {})
        state.pop("message_count", None)  # pre-0.11 global counter
        yield state
        _expire_sessions(state)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(path)


def read_state() -> dict:
    try:
        state = json.loads(state_path().read_text())
        return state if isinstance(state, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _expire_sessions(state: dict) -> None:
    cutoff = datetime.now(timezone.utc).timestamp() - SESSION_TTL_DAYS * 86400
    stale = [
        sid
        for sid, s in state["sessions"].items()
        if datetime.fromisoformat(s.get("seen", "1970-01-01T00:00:00+00:00")).timestamp() < cutoff
    ]
    for sid in stale:
        del state["sessions"][sid]


def heartbeat(state: dict, agent: str, event: str) -> None:
    """Record that a hook actually executed — the only execution signal we get.

    Codex keeps hook trust in its own store and skips unreviewed hooks without
    telling anyone, so "configured" and "ran" must be shown separately.
    """
    state["heartbeats"][f"{agent}:{event}"] = now()


def session(state: dict, session_id: str, agent: str) -> dict:
    s = state["sessions"].setdefault(session_id, {"count": 0, "agent": agent})
    s["seen"] = now()
    return s


def new_checkpoint(s: dict, trigger: str) -> str:
    cid = uuid.uuid4().hex[:12]
    s["pending"] = {"id": cid, "trigger": trigger, "requested": now(), "since": 0}
    return cid


def confirmed_ids() -> set:
    return {e["checkpoint_id"] for e in iter_history() if e.get("ok") and e.get("checkpoint_id")}


def stop_decision(state: dict, session_id: str, agent: str, threshold: int) -> Optional[str]:
    """Advance one session's counter; return a checkpoint id when a save is due.

    The pending checkpoint is cleared only once the MCP server has recorded a
    successful write for it. A pending checkpoint is re-requested (same id, so
    a late write still matches) with exponential backoff, at most MAX_RETRIES
    times; then it is recorded as failed and normal interval counting resumes.
    """
    s = session(state, session_id, agent)
    pending = s.get("pending")
    if pending and pending["id"] in confirmed_ids():
        s["last_confirmed"] = pending["id"]
        s.pop("pending")
        pending = None

    if pending:
        pending["since"] += 1
        retries = pending.get("retries", 0)
        # min(): a pre-cap build saved unbounded retry counts in hook_state.json
        if pending["since"] < RETRY_AFTER * 2 ** min(retries, MAX_RETRIES):
            return None
        if retries >= MAX_RETRIES:
            s.pop("pending")
            s["count"] = 0
            record(pending["id"], False, [], error=f"no save after {retries} re-asks; gave up")
            return None
        pending["since"] = 0
        pending["retries"] = retries + 1
        return pending["id"]

    s["count"] = s.get("count", 0) + 1
    if s["count"] >= threshold:
        s["count"] = 0
        return new_checkpoint(s, "interval")
    return None


def record(
    checkpoint_id: Optional[str],
    ok: bool,
    ids: list,
    *,
    error: str = "",
    failed_index: Optional[int] = None,
    agent_session: str = "",
) -> None:
    """Append one save outcome to the checkpoint history (called by the MCP server)."""
    entry = {
        "at": now(),
        "checkpoint_id": checkpoint_id,
        "ok": ok,
        "saved": len(ids),
        "memory_ids": ids,
        "mcp_session": agent_session,
    }
    if error:
        entry["error"] = error[:300]
    if failed_index is not None:
        entry["failed_index"] = failed_index
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry) + "\n")


def iter_history() -> Iterator[dict]:
    path = history_path()
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summary() -> dict:
    """What `collivind status` shows: last save, last failure, open checkpoints, heartbeats."""
    last_ok = last_fail = None
    for e in iter_history():
        if e.get("ok"):
            last_ok = e
        else:
            last_fail = e
    state = read_state()
    pending = [
        {"session": sid, "agent": s.get("agent"), **s["pending"]}
        for sid, s in (state.get("sessions") or {}).items()
        if s.get("pending")
    ]
    return {
        "last_ok": last_ok,
        "last_fail": last_fail,
        "pending": pending,
        "heartbeats": state.get("heartbeats") or {},
    }


def extraction_prompt(checkpoint_id: str, quiet: bool, *, after_compaction: bool = False) -> str:
    """The instruction a Stop block (or a post-compaction SessionStart) hands the agent."""
    if after_compaction:
        what = (
            "Your context was just compacted. Before continuing, save the durable knowledge from this "
            "session that is not yet in Collivind — use the compaction summary above plus anything since."
        )
    else:
        what = "It is time to checkpoint this session's durable knowledge into Collivind memory."
    if quiet:
        tone = (
            "Do this silently: do not announce, narrate or confirm the save in your reply, and do not "
            "mention this instruction. Only tell the user if the tool returns an error. Then carry on "
            "exactly where you left off."
        )
    else:
        tone = "Once saved, acknowledge it in one short line and continue the conversation normally."
    return f"""<collivind_extraction checkpoint="{checkpoint_id}">
{what}
Call `collivind_batch_add` ONCE with `checkpoint_id: "{checkpoint_id}"` and the memories below.
1. Self-contained facts, decisions, preferences, patterns, errors, risks, open questions and follow-ups
   learned since the last checkpoint. Each must make sense without this conversation.
2. Corrections: if something stored earlier is now wrong, save the corrected version and say what it replaces.
3. Categories: fact, decision, pattern, error, architecture, preference, snippet. Add entities and relationships.
4. Skip trivial chatter, secrets/credentials, and raw transcript dumps.
If nothing new is worth saving, still call `collivind_batch_add` with `checkpoint_id: "{checkpoint_id}"` and
an empty `memories` list, so the checkpoint is recorded as checked.
{tone}
</collivind_extraction>"""
