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


def install_hooks(enable_stop: bool = True, enable_precompact: bool = True,
                  save_interval: int = 15) -> list[str]:
    """Register collivind hooks in ~/.claude/settings.json. Idempotent.

    Replaces any existing collivind entries (so threshold changes apply),
    preserves everything else in the file. Returns the list of events registered.
    """
    settings_path = get_claude_settings_path()
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            raise click.ClickException(
                f"{settings_path} is not valid JSON; fix it manually and re-run."
            )

    wanted = {}
    if enable_stop:
        wanted["Stop"] = f"collivind hook stop --threshold {save_interval}"
    if enable_precompact:
        wanted["PreCompact"] = "collivind hook precompact"

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
    return list(wanted)

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

@hook.command()
def install():
    """Register collivind hooks in ~/.claude/settings.json."""
    from collivind.config import load_config

    cfg = load_config().hooks
    events = install_hooks(cfg.enable_stop, cfg.enable_precompact, cfg.save_interval)
    if events:
        click.secho(f"Registered hooks: {', '.join(events)}", fg="green")
    else:
        click.secho("All hooks disabled in config; nothing registered.", fg="yellow")


@hook.command()
def precompact():
    """Urgent pre-compact extraction hook."""
    click.echo(PRECOMPACT_PROMPT)
    # Reset count since we are extracting now
    state = read_state()
    state["message_count"] = 0
    write_state(state)
