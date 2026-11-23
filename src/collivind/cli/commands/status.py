import json
import subprocess
from pathlib import Path

import click

from collivind.config import load_config
from collivind.docker.health import check_all_services

DOCKER_SERVICES = ("qdrant", "neo4j", "embeddings")


@click.command()
def status():
    """Check health status of Collivind services.

    Exits 1 when a backend is down, a container is crash-looping, or the
    Claude Code MCP server is dead or registered twice — so it can gate a
    script, a cron job, or CI rather than only being read by eye.
    """
    config = load_config()

    click.echo(f"Mode: {config.mode}")
    click.echo("Checking Collivind services status...\n")

    if config.mode == "embedded":
        healthy = _status_embedded(config)
    else:
        healthy = _status_docker(config)
        if config.mode == "docker":
            healthy = _status_containers(config.docker.compose_project) and healthy

    _status_hooks()
    _status_capture()
    healthy = _status_claude_mcp() and healthy
    _status_codex_mcp()
    _status_other_agent_mcp()

    if not healthy:
        click.secho("\nUnhealthy — see the ✗ lines above.", fg="red")
        raise SystemExit(1)


def _report(name: str, health: dict) -> bool:
    ok = health["status"] == "ok"
    click.secho(f"{'✓' if ok else '✗'} {name}: {health['message']}", fg="green" if ok else "red")
    return ok


def _status_embedded(config) -> bool:
    """Check health of embedded backends."""
    from collivind.storage.factory import create_all_backends

    try:
        vector_store, graph_store, embedding_provider = create_all_backends(config)

        results = [
            _report("vector_store", vector_store.health_check()),
            _report("graph_store", graph_store.health_check()),
            _report("embedding_provider", embedding_provider.health_check()),
        ]

        legacy = config.expanded_data_dir / "qdrant_data"
        if legacy.exists():
            click.echo(f"Note: {legacy} is a pre-migration backup of the old vector store; it can be deleted.")
        return all(results)
    except Exception as e:
        click.secho(f"✗ Failed to create backends: {e}", fg="red")
        return False


def _status_docker(config) -> bool:
    """Check health of Docker-based services."""
    return all([_report(service, info) for service, info in check_all_services(config).items()])


def _container_states(names: list[str]) -> list[dict] | None:
    """`docker inspect` state for each container, or None if Docker is unreachable."""
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--format", "{{json .}}", *names],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # inspect exits 1 when any name is missing but still prints the rest.
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _status_containers(project: str) -> bool:
    """Report container state, which an endpoint probe cannot see (#39).

    A crash-looping container answers no probe, so the endpoint check says
    "connection refused" — true but unhelpful. Its own state says *why*:
    restarting, how many times, and Docker's healthcheck verdict.
    """
    names = [f"{project}-{s}" for s in DOCKER_SERVICES]
    states = _container_states(names)
    click.echo("\nContainers:")
    if states is None:
        click.secho("✗ docker: not reachable — start your Docker daemon (e.g. `colima start`)", fg="red")
        return False

    found = {c["Name"].lstrip("/"): c for c in states}
    healthy = True
    for name in names:
        c = found.get(name)
        if c is None:
            click.secho(f"✗ {name}: no such container — run `collivind docker up`", fg="red")
            healthy = False
            continue
        problem = container_problem(c)
        if problem:
            click.secho(f"✗ {name}: {problem}", fg="red")
            healthy = False
        else:
            click.secho(f"✓ {name}: {c['State'].get('Status')}", fg="green")
    return healthy


def container_problem(c: dict) -> str:
    """Why an inspected container is unhealthy, or "" if it is fine."""
    state = c.get("State", {})
    restarts = c.get("RestartCount", 0)
    name = c.get("Name", "").lstrip("/")
    logs = f"see `docker logs --tail 50 {name}`"
    if state.get("Restarting") or (state.get("Status") == "exited" and restarts):
        return f"crash-looping ({restarts} restarts, exit code {state.get('ExitCode')}) — {logs}"
    if state.get("Status") != "running":
        return f"{state.get('Status')} — run `collivind docker up`"
    health = (state.get("Health") or {}).get("Status")
    if health == "unhealthy":
        return f"running but its healthcheck reports unhealthy — {logs}"
    return ""


def _status_claude_mcp() -> bool:
    """Report Claude Code MCP registration (#38): missing, dead, or duplicated.

    Returns False only for states that are broken now (dead or duplicated);
    "not registered" is a setup gap reported as a hint.
    """
    if not (Path.home() / ".claude").exists():
        return True  # not a Claude Code user; nothing to report

    click.echo("\nClaude Code MCP (checking — connects to every configured server)...")
    servers, reason = _claude_mcp_listing()
    if servers is None:
        click.secho(f"? claude MCP: could not check ({reason})", fg="yellow")
        return True

    ours = {name: st for name, st in servers.items() if name == "collivind" or name.endswith(":collivind")}
    if not ours:
        click.secho("✗ claude MCP: no collivind server registered — run `collivind init`", fg="yellow")
        return True

    from collivind.cli.commands.init import _is_healthy

    if "collivind" in ours and len(ours) > 1:
        click.secho(
            f"✗ claude MCP: registered twice ({', '.join(sorted(ours))}) — every tool appears twice in "
            "context. Keep the plugin one: `claude mcp remove -s user collivind`",
            fg="red",
        )
        return False
    dead = {n: st for n, st in ours.items() if not _is_healthy(st)}
    if dead:
        for name, st in dead.items():
            click.secho(
                f"✗ claude MCP: {name} registered but not answering ({st}) — "
                "`claude mcp remove -s user collivind && collivind init`",
                fg="red",
            )
        return False
    click.secho(f"✓ claude MCP: {', '.join(sorted(ours))}", fg="green")
    return True


def _claude_mcp_listing():
    from collivind.cli.commands.init import claude_mcp_servers

    return claude_mcp_servers()


def _status_hooks():
    """Report recall-hook registration per agent.

    The hooks fail silently by design, so without this a broken or missing
    registration looks identical to "nothing worth recalling".
    """
    from collivind.cli.commands.hook import COPILOT_PREVIEW, copilot_warnings, hook_health

    click.echo("\nRecall hooks:")
    for h in hook_health():
        if h["tool"] == "copilot":
            _status_copilot_hooks(h, COPILOT_PREVIEW, copilot_warnings())
        elif not h["events"]:
            why = "no collivind hooks registered" if h["config_exists"] else f"{h['path']} not found"
            click.secho(f"✗ {h['tool']}: {why} — run `collivind hook install`", fg="yellow")
        elif h["source"] == "both":
            # Registered twice: every event fires once per source.
            click.secho(
                f"✗ {h['tool']}: {', '.join(h['events'])} registered BOTH by the plugin and in "
                f"{h['path']} — every hook fires twice. Remove the collivind entries from that file.",
                fg="red",
            )
        elif h["source"] == "plugin":
            click.secho(f"✓ {h['tool']}: {', '.join(h['events'])} (via plugin)", fg="green")
        elif h["broken"]:
            events = ", ".join(h["broken"])
            click.secho(
                f"✗ {h['tool']}: {events} registered but the command no longer resolves"
                " — re-run `collivind hook install`",
                fg="red",
            )
        else:
            click.secho(f"✓ {h['tool']}: {', '.join(h['events'])}", fg="green")


def _status_copilot_hooks(h: dict, preview: str, warnings: list[str]) -> None:
    if not h["events"]:
        click.secho(
            "✗ copilot: no collivind hooks registered — run `collivind hook install --tool copilot`", fg="yellow"
        )
    elif h["broken"]:
        click.secho(
            f"✗ copilot: {', '.join(h['broken'])} registered but the command no longer resolves"
            " — re-run `collivind hook install --tool copilot`",
            fg="red",
        )
    else:
        where = {"workspace": " (workspace .github/hooks only)", "both": " (user and workspace)"}.get(h["source"], "")
        click.secho(f"✓ copilot: {', '.join(h['events'])}{where}", fg="green")
    for warning in warnings:
        click.secho(f"✗ copilot: {warning}", fg="red")
    if h["events"]:
        click.echo(f"  note: {preview}")


def _status_capture():
    """Automatic capture, stated as what actually happened (#35).

    Configured is not trusted, trusted is not executed, and a save request is
    not a save. Hooks leave heartbeats; the MCP server records confirmed
    writes; so each of those states is shown separately rather than implied
    by backend health.
    """
    from collivind import capture
    from collivind.cli.commands.hook import CODEX_REVIEW, hook_health

    info = capture.summary()
    beats = info["heartbeats"]
    click.echo("\nAutomatic capture:")
    for h in hook_health():
        if not h["events"]:
            continue
        ran = {e: beats.get(f"{h['tool']}:{e}") for e in h["events"]}
        never = [e for e, at in ran.items() if not at]
        if never and h["tool"] == "codex":
            click.secho(
                f"? codex: {', '.join(never)} configured but never ran — trust unknown. {CODEX_REVIEW}", fg="yellow"
            )
        elif never:
            click.secho(f"? {h['tool']}: {', '.join(never)} configured but has not run yet", fg="yellow")
        if "Stop" in h["events"] and ran.get("Stop"):
            click.secho(f"✓ {h['tool']}: Stop last ran {ran['Stop']}", fg="green")
    if "codex" in {h["tool"] for h in hook_health() if h["events"]}:
        _warn_codex_inline_hooks()

    ok, fail = info["last_ok"], info["last_fail"]
    if ok:
        what = f"{ok['saved']} memories" if ok["saved"] else "checked, nothing new"
        click.secho(f"✓ last confirmed save: {ok['at']} ({what})", fg="green")
    else:
        click.secho("? no confirmed save recorded yet", fg="yellow")
    if fail and (not ok or fail["at"] > ok["at"]):
        click.secho(f"✗ last save FAILED {fail['at']}: {fail.get('error', '')}", fg="red")
    for p in info["pending"]:
        retries = f", re-asked {p['retries']}x" if p.get("retries") else ""
        click.secho(
            f"? {p.get('agent')} session {p['session'][:12]}: checkpoint {p['id']} ({p['trigger']}) requested "
            f"{p['requested']}, not yet confirmed{retries}",
            fg="yellow",
        )


def _warn_codex_inline_hooks():
    """Codex also reads `[hooks]` inline in config.toml; collivind entries in both fire twice."""
    import tomllib

    cfg = Path.home() / ".codex" / "config.toml"
    try:
        inline = tomllib.loads(cfg.read_text()).get("hooks") or {}
    except (OSError, tomllib.TOMLDecodeError):
        return
    events = sorted(
        ev
        for ev, groups in inline.items()
        if isinstance(groups, list)
        and any("collivind hook" in str(h.get("command", "")) for g in groups for h in g.get("hooks", []))
    )
    if events:
        click.secho(
            f"✗ codex: {', '.join(events)} also declared inline in {cfg} — they fire twice. "
            "Remove the collivind entries from its [hooks] table.",
            fg="red",
        )


def _status_codex_mcp():
    """Report whether Codex can see the MCP tools.

    Hooks and MCP are registered separately, so Codex can have working recall
    hooks and still expose no collivind tools at all (#13).
    """
    import tomllib
    from pathlib import Path

    codex = Path.home() / ".codex"
    if not codex.exists():
        return  # not a Codex user; nothing to report

    config_path = codex / "config.toml"
    registered = False
    if config_path.exists():
        try:
            registered = "collivind" in tomllib.loads(config_path.read_text()).get("mcp_servers", {})
        except tomllib.TOMLDecodeError:
            click.secho(f"✗ codex MCP: {config_path} is not valid TOML", fg="red")
            return

    if registered:
        click.secho("✓ codex MCP: registered", fg="green")
    else:
        click.secho("✗ codex MCP: not registered — run `collivind init`", fg="yellow")


def _status_other_agent_mcp():
    """Report MCP registration for opencode and Copilot (VS Code, CLI).

    Their hooks are reported under "Recall hooks"; this is only whether the
    tools are visible, which is registered separately (#21).
    """
    import json
    from pathlib import Path

    from collivind.cli.commands.hook import copilot_cli_installed, copilot_home
    from collivind.cli.commands.init import vscode_user_dir

    checks: tuple[tuple[str, Path, Path, str], ...] = (
        ("opencode", Path.home() / ".config" / "opencode", Path.home() / ".config/opencode/opencode.json", "mcp"),
        ("copilot", Path.home() / ".vscode", Path.home() / ".vscode/mcp.json", "servers"),
        ("VS Code", vscode_user_dir(), vscode_user_dir() / "mcp.json", "servers"),
    )
    if copilot_cli_installed():
        checks += (("copilot CLI", copilot_home(), copilot_home() / "mcp-config.json", "mcpServers"),)
    for label, agent_dir, config_path, container in checks:
        if not agent_dir.exists():
            continue  # agent not installed; stay quiet
        registered = False
        if config_path.exists():
            try:
                registered = "collivind" in (json.loads(config_path.read_text() or "{}").get(container) or {})
            except json.JSONDecodeError:
                click.secho(f"✗ {label} MCP: {config_path} is not valid JSON", fg="red")
                continue
        if registered:
            click.secho(f"✓ {label} MCP: registered", fg="green")
        else:
            click.secho(f"✗ {label} MCP: not registered — run `collivind init`", fg="yellow")
