import json
from pathlib import Path

import click

EXTRACTION_PROMPT = """
<collivind_extraction>
You are connected to the Collivind memory graph. It is time to extract and save important knowledge from this session so far.
Please use the `collivind_batch_add` tool to store the following:
1. Noteworthy facts, decisions, patterns, errors, or architecture choices you have learned or made.
2. Ensure each memory is self-contained and makes sense without the conversation context.
3. Classify each memory into one of: fact, decision, pattern, error, architecture, preference, snippet.
4. Identify all relevant entities (project, file, service, concept, person, library, tool).
5. Identify any relationships between entities or to previous memories if applicable.
6. Skip trivial, common-sense, or highly ephemeral facts. Focus on knowledge valuable for future sessions.

Once you have successfully invoked the tool and saved the memories, acknowledge it briefly and continue the conversation normally.
</collivind_extraction>
"""

PRECOMPACT_PROMPT = """
<collivind_urgent_extraction>
WARNING: Your context window is about to be compressed or cleared!
You MUST extract ALL relevant facts, context, architecture details, and project state from the current session immediately.
Use the `collivind_batch_add` tool to store everything that might be useful when you resume.
- Be aggressive in saving context.
- Classify correctly (fact, decision, pattern, error, architecture, preference, snippet).
- Extract entities and relationships.

Save the data using the tool right now.
</collivind_urgent_extraction>
"""

HOOK_MARKER = "collivind hook"


def get_state_file() -> Path:
    return Path.home() / ".collivind" / "hook_state.json"


def get_claude_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def get_codex_hooks_path() -> Path:
    return Path.home() / ".codex" / "hooks.json"


def _merge_hook_entries(settings_path: Path, wanted: dict) -> None:
    """Merge collivind hook commands into a hooks JSON file. Idempotent:
    replaces existing collivind entries, preserves everything else."""
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            raise click.ClickException(
                f"{settings_path} is not valid JSON; fix it manually and re-run."
            )

    hooks = settings.setdefault("hooks", {})
    for event, command in wanted.items():
        entries = [
            e for e in hooks.get(event, [])
            if not any(HOOK_MARKER in h.get("command", "") for h in e.get("hooks", []))
        ]
        entries.append({"hooks": [{"type": "command", "command": command}]})
        hooks[event] = entries

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def install_hooks(enable_stop: bool = True, enable_precompact: bool = True,
                  save_interval: int = 15, enable_session_start: bool = True,
                  tool: str = "claude") -> list[str]:
    """Register collivind hooks for a tool. Returns the events registered.

    Claude Code gets all hooks; Codex gets SessionStart only — its
    additionalContext support is documented, Stop/PreCompact injection isn't.
    """
    if tool == "codex":
        wanted = {"SessionStart": "collivind hook session-start"} if enable_session_start else {}
        settings_path = get_codex_hooks_path()
    else:
        wanted = {}
        if enable_stop:
            wanted["Stop"] = f"collivind hook stop --threshold {save_interval}"
        if enable_precompact:
            wanted["PreCompact"] = "collivind hook precompact"
        if enable_session_start:
            wanted["SessionStart"] = "collivind hook session-start"
        settings_path = get_claude_settings_path()

    if wanted:
        _merge_hook_entries(settings_path, wanted)
    return list(wanted)


def install_all_hooks(cfg) -> dict:
    """Install hooks for Claude Code, plus Codex when ~/.codex exists."""
    results = {"claude": install_hooks(cfg.enable_stop, cfg.enable_precompact,
                                       cfg.save_interval, cfg.enable_session_start)}
    if get_codex_hooks_path().parent.exists():
        results["codex"] = install_hooks(cfg.enable_stop, cfg.enable_precompact,
                                         cfg.save_interval, cfg.enable_session_start,
                                         tool="codex")
    return results

def read_state() -> dict:
    state_file = get_state_file()
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"message_count": 0}

def write_state(state: dict):
    state_file = get_state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f)

@click.group()
def hook():
    """Claude Code hook commands."""
    pass

@hook.command()
@click.option('--threshold', default=15, help="Number of messages before extraction")
def stop(threshold):
    """Periodic stop hook."""
    state = read_state()
    count = state.get("message_count", 0) + 1

    if count >= threshold:
        # Stop hooks only reach the model via a block decision; plain stdout
        # is discarded by Claude Code.
        click.echo(json.dumps({"decision": "block", "reason": EXTRACTION_PROMPT}))
        state["message_count"] = 0
    else:
        state["message_count"] = count

    write_state(state)

def _load_manager():
    from collivind.cli.commands.memory import _manager
    return _manager()


@hook.command(name="session-start")
@click.option("--project", "-p", default="default")
@click.option("--limit", "-l", default=8, help="Memories in the index")
def session_start(project, limit):
    """Emit a compact memory index as SessionStart context. Never fails."""
    try:
        memories = _load_manager().get_timeline(project, limit=limit)
    except Exception:
        return  # a broken backend must never break session start
    if not memories:
        return

    lines = [f"- [{m.category.value if hasattr(m.category, 'value') else m.category}] "
             f"{(m.summary or m.content)[:90]}" for m in memories]
    text = (
        "<collivind_memory_index>\n"
        f"Stored knowledge for project '{project}' (most recent first):\n"
        + "\n".join(lines)
        + "\nUse collivind_search or collivind_get_context to recall details.\n"
        "</collivind_memory_index>"
    )
    click.echo(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": text,
        }
    }))


@hook.command()
@click.option("--tool", type=click.Choice(["claude", "codex", "auto"]), default="auto",
              help="Which agent to register hooks for (auto = Claude Code + Codex if present)")
def install(tool):
    """Register collivind hooks (Claude Code settings.json / Codex hooks.json)."""
    from collivind.config import load_config

    cfg = load_config().hooks
    if tool == "auto":
        results = install_all_hooks(cfg)
    else:
        results = {tool: install_hooks(cfg.enable_stop, cfg.enable_precompact,
                                       cfg.save_interval, cfg.enable_session_start, tool=tool)}

    registered = {t: ev for t, ev in results.items() if ev}
    if not registered:
        click.secho("All hooks disabled in config; nothing registered.", fg="yellow")
        return
    for t, events in registered.items():
        click.secho(f"{t}: registered {', '.join(events)}", fg="green")


@hook.command()
def precompact():
    """Urgent pre-compact extraction hook."""
    click.echo(PRECOMPACT_PROMPT)
    # Reset count since we are extracting now
    state = read_state()
    state["message_count"] = 0
    write_state(state)
