import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import click

from collivind.config import generate_default_config, load_config
from collivind.docker.compose import check_docker_running, copy_templates, docker_compose_up
from collivind.docker.health import check_all_services


@click.command()
def init():
    """Initialize Collivind: setup storage backends and start services."""
    click.echo("Initializing Collivind...")

    config = load_config()
    data_dir = config.expanded_data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    config_path = data_dir / "config.toml"
    if not config_path.exists():
        generate_default_config(config_path, config.mode)
        click.echo(f"✓ Config written to {config_path}")

    if config.mode == "embedded":
        _init_embedded(config, data_dir)
    elif config.mode == "remote":
        _init_remote(config)
    else:
        _init_docker(config, data_dir)

    _register_mcp_server()
    _register_hooks(config)
    _register_commands()


def _register_hooks(config):
    """Register agent hooks (Claude Code always, Codex when present)."""
    from collivind.cli.commands.hook import install_all_hooks

    click.echo("Registering agent hooks... ", nl=False)
    try:
        results = {t: ev for t, ev in install_all_hooks(config.hooks).items() if ev}
        if results:
            click.secho("done (" + "; ".join(f"{t}: {', '.join(ev)}" for t, ev in results.items()) + ")", fg="green")
        else:
            click.secho("skipped (disabled in config)", fg="yellow")
    except Exception as e:
        click.secho(f"skipped ({e})", fg="yellow")
        click.echo("Run manually: collivind hook install")


def _register_commands():
    """Install /mem-save and /mem-recall agent commands."""
    from collivind.cli.commands.agent_commands import install_all_commands

    click.echo("Installing agent commands... ", nl=False)
    try:
        results = install_all_commands()
        click.secho(f"done ({', '.join(results)})", fg="green")
    except Exception as e:
        click.secho(f"skipped ({e})", fg="yellow")
        click.echo("Run manually: collivind commands install")


def _init_embedded(config, data_dir: Path):
    """Initialize embedded mode: local storage, no Docker required."""
    click.echo("Mode: embedded (no Docker required)")
    click.echo(f"✓ Data directory: {data_dir}")

    click.echo("Initializing SQLite graph store... ", nl=False)
    try:
        from collivind.storage.graph_sqlite import SqliteGraphStore

        graph = SqliteGraphStore(data_dir=str(data_dir))
        graph.initialize()
        graph.close()
        click.secho("done", fg="green")
    except Exception as e:
        click.secho(f"failed: {e}", fg="red")
        return

    click.echo("Initializing embedded vector store... ", nl=False)
    try:
        from collivind.storage.qdrant_embedded import EmbeddedQdrantStore

        qdrant = EmbeddedQdrantStore(
            data_dir=str(data_dir), config=config.qdrant, dimension=config.embeddings.dimension
        )
        qdrant.initialize()
        qdrant.close()
        click.secho("done", fg="green")
    except Exception as e:
        click.secho(f"failed: {e}", fg="red")
        return

    click.echo("Checking embedding model... ", nl=False)
    try:
        from collivind.storage.embedding_local import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(config.embeddings)
        health = provider.health_check()
        if health["status"] == "ok":
            click.secho("done", fg="green")
        else:
            click.secho(f"warning: {health['message']}", fg="yellow")
            click.echo("Model will be downloaded on first use.")
    except ImportError:
        click.secho("not installed", fg="yellow")
        click.echo("Install with: pip install collivind-memory[embedded]")

    click.secho("\nCollivind is ready (embedded mode).", fg="green")


def _init_remote(config):
    """Initialize remote mode: verify external service connectivity."""
    click.echo("Mode: remote (external services)")

    status = check_all_services(config)
    all_ok = True
    for service, info in status.items():
        if info["status"] == "ok":
            click.secho(f"✓ {service}: {info['message']}", fg="green")
        else:
            click.secho(f"✗ {service}: {info['message']}", fg="red")
            all_ok = False

    if all_ok:
        click.secho("\nCollivind is ready (remote mode).", fg="green")
    else:
        click.secho("\nSome services are not reachable. Check your config.", fg="red")


def _services_already_running(config) -> bool:
    """True when the configured Qdrant/Neo4j/embeddings endpoints all answer.

    collivind may run inside a container that already provides them (an agent
    image, a devcontainer, CI). Provisioning from there is impossible — no
    daemon — and unnecessary (#17).
    """
    try:
        return all(s["status"] == "ok" for s in check_all_services(config).values())
    except Exception:
        return False  # unreachable probe means not usable; fall through and provision


def _init_docker(config, data_dir: Path):
    """Initialize Docker mode: use the configured Qdrant/Neo4j if they already
    answer, otherwise start containers."""
    if _services_already_running(config):
        click.secho("✓ Using already-running Qdrant/Neo4j — skipping container setup", fg="green")
        click.echo(f"  qdrant: {config.qdrant.host}:{config.qdrant.port}")
        click.echo(f"  neo4j:  {config.neo4j.uri}")
        click.secho("Collivind is ready.", fg="green")
        return

    try:
        check_docker_running()
        click.echo("✓ Docker is running")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        return

    click.echo(f"Setting up templates in {data_dir}...")
    try:
        copy_templates(data_dir, config)
        click.echo("✓ Templates generated")
    except Exception as e:
        click.secho(f"Error copying templates: {e}", fg="red")
        return

    click.echo("Starting Docker containers (this may take a few minutes)...")
    try:
        docker_compose_up(data_dir)
        click.echo("✓ Containers started")
    except Exception as e:
        click.secho(f"Error starting containers: {e}", fg="red")
        return

    click.echo("Waiting for services to become healthy...")
    max_retries = 30
    for i in range(max_retries):
        status = check_all_services(config)
        all_ok = all(s["status"] == "ok" for s in status.values())
        if all_ok:
            click.echo("\n✓ All services are healthy!")
            click.secho("Collivind is ready.", fg="green")
            return

        click.echo(".", nl=False)
        time.sleep(2)

    click.secho("\nTimeout waiting for services to become healthy.", fg="red")
    click.echo("Run 'collivind status' to see current state.")


def _mcp_interpreter() -> str:
    """Absolute interpreter that can import collivind.

    Agents spawn MCP servers with their own environment, so a bare `python3`
    may resolve to one without collivind installed — the same class of
    failure as the bare `collivind` in the hook commands (#4).
    """
    return sys.executable or "python3"


def register_codex_mcp() -> bool:
    """Add `[mcp_servers.collivind]` to ~/.codex/config.toml. Returns whether
    the file was changed.

    Appends rather than rewrites: that file holds the user's own model,
    approval and MCP settings, and losing them would be worse than the gap
    this closes (#13).
    """
    codex_dir = Path.home() / ".codex"
    if not codex_dir.exists():
        return False  # not a Codex user; do not create the directory

    config_path = codex_dir / "config.toml"
    existing = config_path.read_text() if config_path.exists() else ""
    if existing:
        try:
            if "collivind" in tomllib.loads(existing).get("mcp_servers", {}):
                return False  # already registered
        except tomllib.TOMLDecodeError:
            return False  # hand-mangled; appending blind would compound it

    block = f'\n[mcp_servers.collivind]\ncommand = "{_mcp_interpreter()}"\nargs = ["-m", "collivind.mcp.server"]\n'
    if existing and not existing.endswith("\n"):
        block = "\n" + block
    config_path.write_text(existing + block)
    return True


def _json_register(config_path: Path, container: str, name: str, entry: dict) -> bool:
    """Add `entry` under `container[name]` in a JSON config. Returns whether
    the file changed.

    Every agent config here holds the user's own settings, so this merges into
    the existing document rather than replacing it, and refuses to touch a file
    it cannot parse instead of appending blind.
    """
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text() or "{}")
        except json.JSONDecodeError:
            return False
        if name in (existing.get(container) or {}):
            return False  # already registered

    existing.setdefault(container, {})[name] = entry
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    return True


def register_opencode_mcp() -> bool:
    """Add collivind to ~/.config/opencode/opencode.json.

    opencode's schema differs from the others: the key is `mcp`, entries carry
    `type: local`, and `command` is a list rather than a command plus args.
    """
    opencode_dir = Path.home() / ".config" / "opencode"
    if not opencode_dir.exists():
        return False  # not an opencode user; do not create the directory

    return _json_register(
        opencode_dir / "opencode.json",
        "mcp",
        "collivind",
        {
            "type": "local",
            "command": [_mcp_interpreter(), "-m", "collivind.mcp.server"],
            "enabled": True,
        },
    )


def register_copilot_mcp() -> bool:
    """Add collivind to VS Code's user-level MCP config (~/.vscode/mcp.json),
    which is what GitHub Copilot reads. Its key is `servers`, not `mcpServers`.
    """
    vscode_dir = Path.home() / ".vscode"
    if not vscode_dir.exists():
        return False  # VS Code not installed for this user

    return _json_register(
        vscode_dir / "mcp.json",
        "servers",
        "collivind",
        {
            "type": "stdio",
            "command": _mcp_interpreter(),
            "args": ["-m", "collivind.mcp.server"],
        },
    )


def _register_mcp_server():
    """Register Collivind as an MCP server with Claude Code, plus Codex when
    ~/.codex exists."""
    click.echo("Registering MCP server with Claude Code... ", nl=False)
    try:
        subprocess.run(
            ["claude", "mcp", "add", "--global", "collivind", "--", "python3", "-m", "collivind.mcp.server"],
            capture_output=True,
            check=True,
        )
        click.secho("done", fg="green")
    except FileNotFoundError:
        click.secho("skipped (claude CLI not found)", fg="yellow")
    except subprocess.CalledProcessError:
        click.secho("skipped (registration failed)", fg="yellow")
        click.echo("Run manually: claude mcp add --global collivind -- python3 -m collivind.mcp.server")

    if (Path.home() / ".codex").exists():
        click.echo("Registering MCP server with Codex... ", nl=False)
        try:
            changed = register_codex_mcp()
            click.secho("done" if changed else "already registered", fg="green")
        except OSError as e:
            click.secho(f"skipped ({e})", fg="yellow")
            click.echo("Add [mcp_servers.collivind] to ~/.codex/config.toml manually.")

    for label, registrar, hint in (
        ("opencode", register_opencode_mcp, "~/.config/opencode/opencode.json"),
        ("Copilot", register_copilot_mcp, "~/.vscode/mcp.json"),
    ):
        try:
            changed = registrar()
        except OSError as e:
            click.secho(f"Registering MCP server with {label}... skipped ({e})", fg="yellow")
            click.echo(f"Add the collivind entry to {hint} manually.")
            continue
        if changed:
            click.secho(f"Registering MCP server with {label}... done", fg="green")
