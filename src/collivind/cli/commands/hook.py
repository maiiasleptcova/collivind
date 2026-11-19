import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import click

from collivind import capture

HOOK_MARKER = "collivind hook"

CODEX_REVIEW = (
    "Codex skips hooks until you trust them: desktop app → Settings → Hooks (or 'Review hooks' in the "
    "composer); CLI → `/hooks`. Re-review after any upgrade that changes the hook commands."
)
CODEX_REVIEW_NEXT_STEP = (
    "Next step for Codex: these hooks will NOT run until you trust them. "
    + CODEX_REVIEW.split(": ", 1)[1]
    + " Restarting Codex does not trust them. Check with `collivind status` after your next Codex message."
)


def get_state_file() -> Path:
    return capture.state_path()


def get_claude_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def get_codex_hooks_path() -> Path:
    return Path.home() / ".codex" / "hooks.json"


def codex_plugin_enabled() -> bool:
    """Whether the Collivind Codex plugin is installed and enabled.

    Codex records plugins in config.toml as `[plugins."<name>@<marketplace>"]`.
    With the plugin providing the hooks (listed under From Plugins), user-level
    entries in hooks.json would run every event twice.
    """
    import tomllib

    try:
        plugins = tomllib.loads((Path.home() / ".codex" / "config.toml").read_text()).get("plugins") or {}
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return any(k.startswith("collivind@") and (v or {}).get("enabled", True) for k, v in plugins.items())


def get_opencode_plugin_path() -> Path:
    return Path.home() / ".config" / "opencode" / "plugins" / "collivind-capture.js"


OPENCODE_EVENTS = ["Stop", "PreCompact"]


def install_opencode_plugin() -> list[str]:
    """Write the opencode capture plugin (opencode has no command hooks)."""
    from importlib.resources import files

    source = files("collivind.integrations").joinpath("opencode-capture.js").read_text()
    path = get_opencode_plugin_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source.replace("__COLLIVIND_BIN__", json.dumps(collivind_bin())))
    return list(OPENCODE_EVENTS)


def collivind_bin() -> str:
    """Absolute path to this collivind executable.

    Agents spawn hooks with their own environment, so a bare `collivind` in
    the hook command silently fails whenever the install dir (uv tools,
    pipx, a venv) isn't on that PATH — the hook never fires and the agent
    gets no memory context. Resolve it at install time instead.
    """
    exe = Path(sys.argv[0])
    if exe.name.startswith("collivind") and exe.is_file():
        return str(exe.resolve())
    return shutil.which("collivind") or "collivind"


def _registered_hooks(settings_path: Path) -> dict:
    """Collivind hook commands in a hooks JSON file, keyed by event."""
    if not settings_path.exists():
        return {}
    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError:
        return {}  # a hand-mangled config must not break `status`
    found = {}
    for event, entries in (settings.get("hooks") or {}).items():
        for entry in entries:
            for h in entry.get("hooks", []):
                if HOOK_MARKER in h.get("command", ""):
                    found[event] = h["command"]
    return found


def _command_resolves(command: str) -> bool:
    """True when the executable a hook command invokes still exists."""
    parts = shlex.split(command)
    if not parts:
        return False
    exe = parts[0]
    if os.sep in exe:
        return os.path.isfile(exe) and os.access(exe, os.X_OK)
    return shutil.which(exe) is not None


def _plugin_hook_events() -> list[str]:
    """Events registered by an installed Claude Code plugin.

    The plugin (#10) ships its own hooks/hooks.json and Claude Code loads them
    from the plugin cache, leaving ~/.claude/settings.json untouched. Reporting
    "none registered" there sent users to `hook install`, which adds a second
    copy so every event fires twice (#23).
    """
    cache = Path.home() / ".claude" / "plugins"
    if not cache.exists():
        return []
    events: set[str] = set()
    for manifest in cache.rglob("hooks/hooks.json"):
        if "collivind" not in str(manifest):
            continue
        try:
            hooks = (json.loads(manifest.read_text()).get("hooks") or {}).items()
        except (OSError, json.JSONDecodeError):
            continue
        for event, entries in hooks:
            for entry in entries:
                for h in entry.get("hooks", []):
                    if HOOK_MARKER in h.get("command", "") or "collivind-run" in h.get("command", ""):
                        events.add(event)
    return sorted(events)


def hook_health() -> list[dict]:
    """Per-agent recall-hook health.

    Both recall hooks swallow every exception on purpose, so a missing
    registration or an uninstalled binary makes memory injection stop
    silently. This is what makes that visible.
    """
    report = []
    plugin_events = _plugin_hook_events()
    for tool, path in (("claude", get_claude_settings_path()), ("codex", get_codex_hooks_path())):
        registered = _registered_hooks(path)
        if tool == "claude":
            from_plugin = plugin_events
        else:
            from_plugin = ["PreCompact", "SessionStart", "Stop", "UserPromptSubmit"] if codex_plugin_enabled() else []

        if registered and from_plugin:
            source = "both"  # a real misconfiguration: every event fires twice
        elif from_plugin:
            source = "plugin"
        elif registered:
            source = "settings"
        else:
            source = "none"

        report.append(
            {
                "tool": tool,
                "path": str(path),
                "config_exists": path.exists(),
                "source": source,
                "events": sorted(set(registered) | set(from_plugin)),
                "broken": sorted(e for e, cmd in registered.items() if not _command_resolves(cmd)),
            }
        )
    plugin = get_opencode_plugin_path()
    if plugin.parent.parent.exists():
        report.append(
            {
                "tool": "opencode",
                "path": str(plugin),
                "config_exists": plugin.exists(),
                "source": "plugin" if plugin.exists() else "none",
                "events": list(OPENCODE_EVENTS) if plugin.exists() else [],
                "broken": [],
            }
        )
    return report


def _merge_hook_entries(settings_path: Path, wanted: dict) -> None:
    """Merge collivind hook commands into a hooks JSON file. Idempotent:
    replaces existing collivind entries, preserves everything else."""
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            raise click.ClickException(f"{settings_path} is not valid JSON; fix it manually and re-run.")

    hooks = settings.setdefault("hooks", {})
    for event, command in wanted.items():
        entries = [
            e for e in hooks.get(event, []) if not any(HOOK_MARKER in h.get("command", "") for h in e.get("hooks", []))
        ]
        entries.append({"hooks": [{"type": "command", "command": command}]})
        hooks[event] = entries

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def install_hooks(
    enable_stop: bool = True,
    enable_precompact: bool = True,
    save_interval: int = 15,
    enable_session_start: bool = True,
    tool: str = "claude",
    enable_user_prompt: bool = True,
) -> list[str]:
    """Register collivind hooks for a tool. Returns the events registered.

    Both agents get recall (SessionStart, UserPromptSubmit) and capture
    (Stop, PreCompact). Codex documents the same Stop `decision: block`
    contract as Claude Code and a PreCompact event (#35); its PreCompact
    cannot inject context, so the save is requested by SessionStart with
    `source: "compact"` right after compaction instead.

    The installed command strings carry no tunables (the interval lives in
    config.toml): Codex trusts hooks by a hash of their definition, so any
    change to the string forces the user to re-review them.
    """
    if tool == "opencode":
        return install_opencode_plugin() if (enable_stop or enable_precompact) else []
    bin_path = collivind_bin()
    agent = "" if tool == "claude" else f" --agent {tool}"
    wanted = {}
    if enable_stop:
        wanted["Stop"] = f"{bin_path} hook stop{agent}"
    if enable_precompact:
        wanted["PreCompact"] = f"{bin_path} hook precompact{agent}"
    if enable_session_start:
        wanted["SessionStart"] = f"{bin_path} hook session-start{agent}"
    if enable_user_prompt:
        wanted["UserPromptSubmit"] = f"{bin_path} hook user-prompt{agent}"
    settings_path = get_codex_hooks_path() if tool == "codex" else get_claude_settings_path()

    if wanted:
        _merge_hook_entries(settings_path, wanted)
    return list(wanted)


def install_all_hooks(cfg) -> dict:
    """Install hooks for Claude Code, plus Codex when ~/.codex exists."""

    def _install(tool):
        return install_hooks(
            cfg.enable_stop,
            cfg.enable_precompact,
            cfg.save_interval,
            cfg.enable_session_start,
            tool=tool,
            enable_user_prompt=cfg.enable_user_prompt,
        )

    results = {"claude": _install("claude")}
    if get_codex_hooks_path().parent.exists() and not codex_plugin_enabled():
        results["codex"] = _install("codex")
    if get_opencode_plugin_path().parent.parent.exists():
        results["opencode"] = _install("opencode")
    return results


@click.group()
def hook():
    """Claude Code hook commands."""
    pass


def _hook_input() -> dict:
    """The JSON payload agents pipe to hooks; {} when there is none."""
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return {}
        data = sys.stdin.read()
        payload = json.loads(data) if data.strip() else {}
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _hooks_config():
    from collivind.config import load_config

    try:
        return load_config().hooks
    except Exception:
        from collivind.config import HooksConfig

        return HooksConfig()


AGENT = click.option(
    "--agent", default="claude", type=click.Choice(["claude", "codex", "opencode"]), help="Which agent runs this hook"
)


@hook.command()
@click.option("--threshold", default=None, type=int, help="Responses between checkpoints (default: config)")
@click.option("--session", "session_id", default=None, help="Session id when the agent passes none on stdin")
@AGENT
def stop(threshold, session_id, agent):
    """Periodic capture checkpoint. Asks the agent to save; never assumes it did."""
    payload = _hook_input()
    cfg = _hooks_config()
    threshold = threshold or cfg.save_interval
    sid = session_id or payload.get("session_id") or "default"
    with capture.locked_state() as state:
        capture.heartbeat(state, agent, "Stop")
        if payload.get("stop_hook_active"):
            return  # this response is our own continuation; never chain blocks
        checkpoint = capture.stop_decision(state, sid, agent, threshold)
    if checkpoint:
        # Stop hooks reach the model only through a block decision; both
        # Claude Code and Codex turn `reason` into a continuation prompt.
        click.echo(
            json.dumps({"decision": "block", "reason": capture.extraction_prompt(checkpoint, cfg.quiet_capture)})
        )


def _load_manager():
    from collivind.cli.commands.memory import _manager

    return _manager()


@hook.command(name="session-start")
@click.option("--project", "-p", default="default")
@click.option("--limit", "-l", default=8, help="Memories in the index")
@AGENT
def session_start(project, limit, agent):
    """Emit a compact memory index as SessionStart context. Never fails.

    After a compaction (`source: "compact"`) it also asks the agent to save
    what the compaction may have dropped — the only point at which Codex lets
    a hook speak to the model around compaction.
    """
    payload = _hook_input()
    save_request = ""
    try:
        with capture.locked_state() as state:
            capture.heartbeat(state, agent, "SessionStart")
            if payload.get("source") == "compact":
                save_request = _compaction_request(state, payload.get("session_id") or "default", agent)
    except OSError:
        pass  # state is best-effort; recall still runs

    try:
        memories = _load_manager().get_timeline(project, limit=limit)
    except Exception as e:
        # Never break session start, but never go silent either: an empty
        # recall looked identical to a dead store for 46 hours in #39.
        _emit_context("SessionStart", _unavailable_warning(e))
        return
    if not memories:
        if save_request:
            _emit_context("SessionStart", save_request)
        return

    lines = [
        f"- [{m.category.value if hasattr(m.category, 'value') else m.category}] {(m.summary or m.content)[:90]}"
        for m in memories
    ]
    text = (
        "<collivind_memory_index>\n"
        f"Stored knowledge for project '{project}' (most recent first):\n"
        + "\n".join(lines)
        + "\nUse collivind_search or collivind_get_context to recall details.\n"
        "</collivind_memory_index>"
    )
    _emit_context("SessionStart", f"{text}\n{save_request}" if save_request else text)


def _compaction_request(state: dict, session_id: str, agent: str) -> str:
    s = capture.session(state, session_id, agent)
    cid = (s.get("pending") or {}).get("id") or capture.new_checkpoint(s, "compaction")
    return capture.extraction_prompt(cid, _hooks_config().quiet_capture, after_compaction=True)


@hook.command()
@click.option("--session", "session_id", required=True)
@AGENT
def compacted(session_id, agent):
    """Print the post-compaction save request as JSON (for agents without SessionStart, e.g. opencode)."""
    with capture.locked_state() as state:
        capture.heartbeat(state, agent, "SessionStart")
        click.echo(json.dumps({"prompt": _compaction_request(state, session_id, agent)}))


def _emit_context(event: str, text: str) -> None:
    click.echo(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))


def _unavailable_warning(error: Exception) -> str:
    first_line = (str(error).strip().splitlines() or [type(error).__name__])[0][:200]
    return (
        "<collivind_warning>\n"
        f"Collivind memory is UNAVAILABLE, so no stored context was recalled: {first_line}\n"
        "Tell the user, and suggest running `collivind status` to see which backend is down.\n"
        "</collivind_warning>"
    )


@hook.command(name="user-prompt")
@click.option("--project", "-p", default="default")
@click.option("--max-tokens", default=800, type=int, help="Recall context budget")
@AGENT
def user_prompt(project, max_tokens, agent):
    """Inject memory context relevant to the submitted prompt. Never fails.

    Reads the UserPromptSubmit payload from stdin, searches memory with the
    prompt text, and emits a token-budgeted context block — or nothing when
    no stored knowledge is relevant.
    """
    try:
        with capture.locked_state() as state:
            capture.heartbeat(state, agent, "UserPromptSubmit")
    except OSError:
        pass
    try:
        payload = json.load(sys.stdin)
        prompt = (payload.get("prompt") or "").strip()
    except Exception:
        return  # malformed payload: nothing to recall against
    if len(prompt) < 15:
        return  # "yes", "ok", etc. — not worth an embedding round-trip

    try:
        manager = _load_manager()
        # Nothing stored for this project means nothing to recall: skip the
        # embedding, which in embedded mode is a cold model load that can blow
        # the hook timeout on a first run (#16). A timeline read is plain SQL.
        if not manager.get_timeline(project, limit=1):
            return
        # ponytail: embedded mode still loads the model per invocation on a
        # populated store; a resident daemon is the real fix (ROADMAP.md, #16)
        context = manager.get_context(prompt, project_id=project, limit=5, max_tokens=max_tokens)
    except Exception as e:
        # Recall must never break prompt submission — but say it failed.
        _emit_context("UserPromptSubmit", _unavailable_warning(e))
        return
    if "No relevant context found" in context:
        return
    _emit_context("UserPromptSubmit", f"<collivind_recall>\n{context}\n</collivind_recall>")


@hook.command()
@click.option(
    "--tool",
    type=click.Choice(["claude", "codex", "opencode", "auto"]),
    default="auto",
    help="Which agent to register hooks for (auto = Claude Code, plus Codex and opencode if present)",
)
def install(tool):
    """Register collivind hooks (Claude Code settings.json / Codex hooks.json)."""
    from collivind.config import load_config

    cfg = load_config().hooks
    if tool == "auto":
        results = install_all_hooks(cfg)
    else:
        results = {
            tool: install_hooks(
                cfg.enable_stop,
                cfg.enable_precompact,
                cfg.save_interval,
                cfg.enable_session_start,
                tool=tool,
                enable_user_prompt=cfg.enable_user_prompt,
            )
        }

    registered = {t: ev for t, ev in results.items() if ev}
    if not registered:
        click.secho("All hooks disabled in config; nothing registered.", fg="yellow")
        return
    for t, events in registered.items():
        click.secho(f"{t}: registered {', '.join(events)}", fg="green")
    if "codex" in registered:
        click.secho(CODEX_REVIEW_NEXT_STEP, fg="yellow")


@hook.command()
@click.option("--session", "session_id", default=None, help="Session id when the agent passes none on stdin")
@AGENT
def precompact(session_id, agent):
    """Mark a compaction checkpoint as due; SessionStart(compact) requests it.

    Neither agent lets PreCompact inject context into the model (Codex
    ignores its stdout), so printing an instruction here was never a save.
    The request is made right after compaction instead, carrying the same
    checkpoint id, and stays pending until a write confirms it.
    """
    payload = _hook_input()
    with capture.locked_state() as state:
        capture.heartbeat(state, agent, "PreCompact")
        s = capture.session(state, session_id or payload.get("session_id") or "default", agent)
        if not s.get("pending"):
            capture.new_checkpoint(s, f"compaction:{payload.get('trigger') or 'unknown'}")
        s["count"] = 0
