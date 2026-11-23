import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

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


# Plugin flag -> the Claude Code event it stands in for (heartbeat names).
OPENCODE_EVENTS = {
    "stop": "Stop",
    "precompact": "PreCompact",
    "sessionStart": "SessionStart",
    "userPrompt": "UserPromptSubmit",
}

# GitHub Copilot (VS Code agent mode, Copilot CLI, Copilot app) event names for
# the agent-neutral ones used in heartbeats and `status`. camelCase keys select
# Copilot's camelCase payload (`sessionId`) and its top-level output contract.
COPILOT_EVENTS = {
    "SessionStart": "sessionStart",
    "UserPromptSubmit": "userPromptTransformed",
    "Stop": "agentStop",
    "PreCompact": "preCompact",
}
COPILOT_TIMEOUTS = {"sessionStart": 30, "userPromptTransformed": 30, "agentStop": 10, "preCompact": 10}
COPILOT_PREVIEW = "Copilot agent hooks are in preview; an organization policy can turn them off."


def copilot_home() -> Path:
    return Path(os.environ.get("COPILOT_HOME") or Path.home() / ".copilot")


def get_copilot_hooks_path() -> Path:
    return copilot_home() / "hooks" / "collivind.json"


def copilot_present() -> bool:
    """Copilot CLI/app (~/.copilot) or VS Code (~/.vscode) is installed."""
    return copilot_home().exists() or (Path.home() / ".vscode").exists()


def repo_root() -> Optional[Path]:
    """The current git repo's root; None outside one, or for a dotfiles repo
    at ~ (whose .claude/ is the user-level config, not a repo's)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    root = Path(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    return None if root is None or root.resolve() == Path.home().resolve() else root


def get_workspace_hooks_path() -> Optional[Path]:
    root = repo_root()
    return root / ".github" / "hooks" / "collivind.json" if root else None


def write_copilot_hooks(path: Path, bin_path: str, subcommands: dict) -> None:
    """Write Copilot's hooks file. collivind owns the whole file, so rewriting
    it is idempotent and leaves the user's other hook files alone."""
    quoted_ps = "& '" + bin_path.replace("'", "''") + "'"
    hooks = {}
    for event, sub in subcommands.items():
        name = COPILOT_EVENTS[event]
        args = f"hook {sub} --agent copilot"
        hooks[name] = [
            {
                "type": "command",
                "bash": f"{shlex.quote(bin_path)} {args}",
                "powershell": f"{quoted_ps} {args}",
                "timeoutSec": COPILOT_TIMEOUTS[name],
            }
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "hooks": hooks}, indent=2) + "\n")


def _copilot_registered(path: Optional[Path]) -> dict:
    """Collivind commands in a Copilot hooks file, keyed by agent-neutral event."""
    if path is None:
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    canonical = {v: k for k, v in COPILOT_EVENTS.items()}
    found = {}
    for event, entries in ((doc.get("hooks") or {}) if isinstance(doc, dict) else {}).items():
        for entry in entries if isinstance(entries, list) else []:
            cmd = (entry.get("bash") or entry.get("command") or "") if isinstance(entry, dict) else ""
            if "--agent copilot" in cmd:
                found[canonical.get(event, event)] = cmd
    return found


def _repo_claude_hooks() -> list[str]:
    """Collivind events in the repo's .claude/settings*.json, which VS Code's
    Copilot harness also runs — alongside the Copilot hooks, so twice."""
    root = repo_root()
    if root is None:
        return []
    return sorted(
        {e for f in ("settings.json", "settings.local.json") for e in _registered_hooks(root / ".claude" / f)}
    )


def copilot_warnings() -> list[str]:
    warnings = []
    user, workspace = get_copilot_hooks_path(), get_workspace_hooks_path()
    if _copilot_registered(user) and _copilot_registered(workspace):
        warnings.append(
            f"collivind hooks are in both {user} and {workspace}; locally every Copilot hook fires twice. "
            "Keep the workspace file only for the cloud agent, or remove one."
        )
    repo = _repo_claude_hooks()
    if repo:
        warnings.append(
            f"The repo's .claude/settings*.json registers collivind {', '.join(repo)} and VS Code's Copilot harness runs "
            "those too, so these events fire twice under Copilot. Move them to ~/.claude/settings.json."
        )
    return warnings


def install_opencode_plugin(enabled: dict) -> list[str]:
    """Write the opencode plugin (opencode has no command hooks).

    The enabled events are baked into the file, as the hooks config gates
    which commands Claude Code and Codex get registered.
    """
    from importlib.resources import files

    source = files("collivind.integrations").joinpath("opencode-capture.js").read_text()
    path = get_opencode_plugin_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = {k: bool(enabled.get(k)) for k in OPENCODE_EVENTS}
    path.write_text(
        source.replace("__COLLIVIND_BIN__", json.dumps(collivind_bin())).replace(
            "__COLLIVIND_ENABLED__", json.dumps(flags)
        )
    )
    return [e for k, e in OPENCODE_EVENTS.items() if flags[k]]


def _opencode_plugin_events(path: Path) -> list[str]:
    """Events an installed opencode plugin runs; plugins from before 0.14 only capture."""
    try:
        line = next(ln for ln in path.read_text().splitlines() if ln.startswith("const ENABLED = "))
        flags = json.loads(line.removeprefix("const ENABLED = ").rstrip(";"))
        return [e for k, e in OPENCODE_EVENTS.items() if flags.get(k)]
    except (OSError, StopIteration, ValueError, AttributeError):
        return ["Stop", "PreCompact"]


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
    user_path, workspace_path = get_copilot_hooks_path(), get_workspace_hooks_path()
    user, workspace = _copilot_registered(user_path), _copilot_registered(workspace_path)
    if copilot_present() or workspace:
        report.append(
            {
                "tool": "copilot",
                "path": str(user_path),
                "config_exists": user_path.exists(),
                "source": "both"
                if user and workspace
                else "settings"
                if user
                else "workspace"
                if workspace
                else "none",
                "events": sorted(set(user) | set(workspace)),
                "broken": sorted(
                    {e for found in (user, workspace) for e, cmd in found.items() if not _command_resolves(cmd)}
                ),
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
                "events": _opencode_plugin_events(plugin) if plugin.exists() else [],
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
    workspace: bool = False,
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
        enabled = {
            "stop": enable_stop,
            "precompact": enable_precompact,
            "sessionStart": enable_session_start,
            "userPrompt": enable_user_prompt,
        }
        return install_opencode_plugin(enabled) if any(enabled.values()) else []
    subcommands = {}
    if enable_stop:
        subcommands["Stop"] = "stop"
    if enable_precompact:
        subcommands["PreCompact"] = "precompact"
    if enable_session_start:
        subcommands["SessionStart"] = "session-start"
    if enable_user_prompt:
        subcommands["UserPromptSubmit"] = "user-prompt"
    if tool == "copilot":
        # A workspace file is committed and read by the cloud agent, where this
        # machine's absolute path means nothing; there it is plain `collivind`.
        if subcommands:
            path = get_workspace_hooks_path() if workspace else get_copilot_hooks_path()
            if path is None:
                raise click.UsageError("--workspace must run inside a git repository")
            write_copilot_hooks(path, "collivind" if workspace else collivind_bin(), subcommands)
        return [COPILOT_EVENTS[e] for e in subcommands]
    bin_path = collivind_bin()
    agent = "" if tool == "claude" else f" --agent {tool}"
    wanted = {e: f"{bin_path} hook {sub}{agent}" for e, sub in subcommands.items()}
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
    if copilot_present():
        results["copilot"] = _install("copilot")
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
    "--agent",
    default="claude",
    type=click.Choice(["claude", "codex", "opencode", "copilot"]),
    help="Which agent runs this hook",
)


@hook.command()
@click.option("--threshold", default=None, type=int, help="Responses between checkpoints (default: config)")
@click.option("--session", "session_id", default=None, help="Session id when the agent passes none on stdin")
@click.option(
    "--continuation", is_flag=True, help="This response answers our own save request (opencode's stop_hook_active)"
)
@AGENT
def stop(threshold, session_id, continuation, agent):
    """Periodic capture checkpoint. Asks the agent to save; never assumes it did."""
    payload = _hook_input()
    cfg = _hooks_config()
    # >= 2: a backstop should an agent ever report its own continuation
    # neither as stop_hook_active nor as --continuation.
    threshold = max(2, threshold or cfg.save_interval)
    sid = session_id or _session_id(payload)
    with capture.locked_state() as state:
        capture.heartbeat(state, agent, "Stop")
        if continuation or payload.get("stop_hook_active"):
            return  # this response is our own continuation; never chain blocks
        checkpoint = capture.stop_decision(state, sid, agent, threshold)
    if checkpoint:
        # Stop hooks reach the model only through a block decision; both
        # Claude Code and Codex turn `reason` into a continuation prompt.
        click.echo(
            json.dumps({"decision": "block", "reason": capture.extraction_prompt(checkpoint, cfg.quiet_capture)})
        )


def _session_id(payload: dict) -> str:
    # Claude Code, Codex and Copilot's PascalCase events send session_id;
    # Copilot's camelCase events send sessionId.
    return payload.get("session_id") or payload.get("sessionId") or "default"


def _load_manager():
    from collivind.cli.commands.memory import _manager

    return _manager()


PROJECT = click.option(
    "--project", "-p", default=None, help="Project id (default: resolved from the session's working directory)"
)


def _recall_projects(project: Optional[str], payload: dict) -> list:
    """The project to recall from, then the unlabelled "default" bucket, unless one was named."""
    from collivind.engine.project_labels import DEFAULT_PROJECT, resolve_project

    if project:
        return [project]
    return list(dict.fromkeys([resolve_project(payload.get("cwd")), DEFAULT_PROJECT]))


@hook.command(name="session-start")
@PROJECT
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
                save_request = _compaction_request(state, _session_id(payload), agent)
    except OSError:
        pass  # state is best-effort; recall still runs

    try:
        manager = _load_manager()
        memories = []
        projects = _recall_projects(project, payload)
        for project in projects:
            memories = manager.get_timeline(project, limit=limit)
            if memories:
                break
    except Exception as e:
        # Never break session start, but never go silent either: an empty
        # recall looked identical to a dead store for 46 hours in #39.
        warning = _unavailable_warning(e)
        _emit_context("SessionStart", f"{warning}\n{save_request}" if save_request else warning, agent)
        return
    if not memories:
        if save_request:
            _emit_context("SessionStart", save_request, agent)
        return

    lines = [
        f"- [{m.category.value if hasattr(m.category, 'value') else m.category}] {(m.summary or m.content)[:90]}"
        for m in memories
    ]
    text = (
        "<collivind_memory_index>\n"
        + (
            f"Stored knowledge for project '{project}' (most recent first):\n"
            if project == projects[0]
            else f"Project '{projects[0]}' has no memories yet; older unlabelled ones (most recent first):\n"
        )
        + "\n".join(lines)
        + "\nUse collivind_search or collivind_get_context to recall details.\n"
        "</collivind_memory_index>"
    )
    _emit_context("SessionStart", f"{text}\n{save_request}" if save_request else text, agent)


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


def _emit_context(event: str, text: str, agent: str = "claude") -> None:
    if agent == "copilot":  # Copilot reads only a top-level additionalContext
        click.echo(json.dumps({"additionalContext": text}))
        return
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
@PROJECT
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
        context = ""
        for project in _recall_projects(project, payload):
            # Nothing stored for this project means nothing to recall: skip the
            # embedding, which in embedded mode is a cold model load that can blow
            # the hook timeout on a first run (#16). A timeline read is plain SQL.
            # If the graph cannot answer, carry on: search degrades to the vector
            # store alone (#39) and says so.
            try:
                if not manager.get_timeline(project, limit=1):
                    continue
            except Exception:
                pass
            context = manager.get_context(prompt, project_id=project, limit=5, max_tokens=max_tokens)
            if "No relevant context found" not in context:
                break
        if not context:
            return
    except Exception as e:
        # Recall must never break prompt submission — but say it failed.
        _emit_prompt_context(agent, payload, _unavailable_warning(e))
        return
    if "No relevant context found" in context and "[DEGRADED]" not in context:
        return
    _emit_prompt_context(agent, payload, f"<collivind_recall>\n{context}\n</collivind_recall>")


def _emit_prompt_context(agent: str, payload: dict, text: str) -> None:
    """Copilot ignores config-hook output on userPromptSubmitted; its
    userPromptTransformed hook may replace the model-facing content instead.
    Prepend to that content and never emit a replacement without it."""
    if agent != "copilot":
        _emit_context("UserPromptSubmit", text)
        return
    base = payload.get("transformedPrompt")
    if isinstance(base, str) and base.strip():
        click.echo(json.dumps({"modifiedTransformedPrompt": f"{text}\n\n{base}"}))


@hook.command()
@click.option(
    "--tool",
    type=click.Choice(["claude", "codex", "opencode", "copilot", "auto"]),
    default="auto",
    help="Which agent to register hooks for (auto = Claude Code, plus Codex, opencode and Copilot if present)",
)
@click.option(
    "--workspace",
    is_flag=True,
    help="Copilot only: write .github/hooks/collivind.json in this repo (what the cloud agent reads)",
)
def install(tool, workspace):
    """Register collivind hooks (Claude Code settings.json / Codex hooks.json / Copilot hooks/collivind.json)."""
    if workspace and tool != "copilot":
        raise click.UsageError("--workspace applies to --tool copilot only")
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
                workspace=workspace,
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
    if "copilot" in registered:
        click.secho(COPILOT_PREVIEW, fg="yellow")
        for warning in copilot_warnings():
            click.secho(f"Warning: {warning}", fg="red")


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
        s = capture.session(state, session_id or _session_id(payload), agent)
        if not s.get("pending"):
            capture.new_checkpoint(s, f"compaction:{payload.get('trigger') or 'unknown'}")
        if agent == "copilot":
            # Copilot never fires sessionStart after compaction, so the next
            # agentStop must ask for this save at once, not after the interval.
            s["pending"]["due"] = True
        s["count"] = 0
