import click

from collivind.config import load_config
from collivind.docker.health import check_all_services


@click.command()
def status():
    """Check health status of Collivind services."""
    config = load_config()

    click.echo(f"Mode: {config.mode}")
    click.echo("Checking Collivind services status...\n")

    if config.mode == "embedded":
        _status_embedded(config)
    else:
        _status_docker(config)

    _status_hooks()
    _status_codex_mcp()
    _status_other_agent_mcp()


def _status_embedded(config):
    """Check health of embedded backends."""
    from collivind.storage.factory import create_all_backends

    try:
        vector_store, graph_store, embedding_provider = create_all_backends(config)

        backends = [
            ("vector_store", vector_store),
            ("graph_store", graph_store),
            ("embedding_provider", embedding_provider),
        ]
        for name, backend in backends:
            health = backend.health_check()
            if health["status"] == "ok":
                click.secho(f"✓ {name}: {health['message']}", fg="green")
            else:
                click.secho(f"✗ {name}: {health['message']}", fg="red")

        legacy = config.expanded_data_dir / "qdrant_data"
        if legacy.exists():
            click.echo(f"Note: {legacy} is a pre-migration backup of the old vector store; it can be deleted.")
    except Exception as e:
        click.secho(f"✗ Failed to create backends: {e}", fg="red")


def _status_docker(config):
    """Check health of Docker-based services."""
    status = check_all_services(config)

    for service, info in status.items():
        if info["status"] == "ok":
            click.secho(f"✓ {service}: {info['message']}", fg="green")
        else:
            click.secho(f"✗ {service}: {info['message']}", fg="red")


def _status_hooks():
    """Report recall-hook registration per agent.

    The hooks fail silently by design, so without this a broken or missing
    registration looks identical to "nothing worth recalling".
    """
    from collivind.cli.commands.hook import hook_health

    click.echo("\nRecall hooks:")
    for h in hook_health():
        if not h["events"]:
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
    """Report MCP registration for opencode and Copilot.

    Neither has a hook system, so they get tools on demand but no automatic
    recall — worth being explicit rather than implying parity with Claude
    Code and Codex (#21).
    """
    import json
    from pathlib import Path

    checks = (
        ("opencode", Path.home() / ".config" / "opencode", Path.home() / ".config/opencode/opencode.json", "mcp"),
        ("copilot", Path.home() / ".vscode", Path.home() / ".vscode/mcp.json", "servers"),
    )
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
            click.secho(f"✓ {label} MCP: registered (tools only — no recall hooks)", fg="green")
        else:
            click.secho(f"✗ {label} MCP: not registered — run `collivind init`", fg="yellow")
