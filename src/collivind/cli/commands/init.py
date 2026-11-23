import json
import os
import shlex
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
            if "codex" in results:
                from collivind.cli.commands.hook import CODEX_REVIEW_NEXT_STEP

                click.secho(CODEX_REVIEW_NEXT_STEP, fg="yellow")
            if "copilot" in results:
                from collivind.cli.commands.hook import copilot_warnings

                for warning in copilot_warnings():
                    click.secho(f"Warning: {warning}", fg="red")
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


def claude_mcp_servers() -> tuple[dict[str, str] | None, str]:
    """`{server name: health status}` Claude Code reports, plus a failure reason.

    Returns `(None, reason)` when the listing could not be read at all. The
    three ways that happens are not interchangeable — an absent CLI, a hung
    one, and a broken config each need a different thing from the user — so
    the reason is carried out rather than collapsed, which is the mistake
    that let `--global` sit unnoticed for as long as it did.

    `claude mcp get <name>` exits 0 even for a server that does not exist, so
    the listing is the only usable signal. Its lines read
    `<name>: <command> - <status>`; a name may contain colons
    (`plugin:collivind:collivind`) but never a colon followed by a space, and
    the status follows the last `" - "`.

    Listing health-checks by connecting to every configured server, so this
    can take seconds on a cold npx cache or a remote endpoint.
    """
    try:
        proc = subprocess.run(
            ["claude", "mcp", "list"],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None, "claude CLI not found"
    except subprocess.TimeoutExpired:
        return None, "claude mcp list timed out after 60s"
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"could not run claude mcp list: {e}"
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        return None, detail[0] if detail else "claude mcp list failed"

    servers = {}
    for line in proc.stdout.splitlines():
        if not line or line[0].isspace() or line[0] in "├└│":
            continue  # continuation lines of a multi-scope warning
        name, sep, rest = line.partition(": ")
        if sep:
            _, _, status = rest.rpartition(" - ")
            servers[name] = status.strip()
    return servers, ""


def _is_healthy(status: str) -> bool:
    """Whether a listing row reports a server that actually answered."""
    low = status.lower()
    return "connected" in low and "failed" not in low


def _collivind_already_registered(names: set[str]) -> str | None:
    """The name Claude Code already serves collivind under, if any.

    The plugin — the install this project recommends — registers its own
    server as `plugin:collivind:collivind`, which does not collide with a
    plain `collivind` add. Adding on top of it would give the user two
    identical servers and two copies of every tool in context.
    """
    return next((n for n in sorted(names) if n == "collivind" or n.endswith(":collivind")), None)


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


def vscode_user_dir() -> Path:
    """VS Code's user profile folder, where it reads the user-level mcp.json."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / "User"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Code" / "User"
    return Path.home() / ".config" / "Code" / "User"


def register_vscode_profile_mcp() -> bool:
    """Add collivind to VS Code's profile mcp.json, so agent mode has the tool
    its agentStop checkpoints ask for."""
    user_dir = vscode_user_dir()
    if not user_dir.exists():
        return False  # VS Code has never run for this user
    return _json_register(
        user_dir / "mcp.json",
        "servers",
        "collivind",
        {"type": "stdio", "command": _mcp_interpreter(), "args": ["-m", "collivind.mcp.server"]},
    )


def register_copilot_cli_mcp() -> bool:
    """Add collivind to the Copilot CLI/app MCP config ($COPILOT_HOME/mcp-config.json).

    Its Stop hook asks the agent to call `collivind_batch_add`; without the
    server registered there the tool does not exist and every save fails.
    """
    from collivind.cli.commands.hook import copilot_cli_installed, copilot_home

    if not copilot_cli_installed():
        return False  # not a Copilot CLI user; do not create the config

    return _json_register(
        copilot_home() / "mcp-config.json",
        "mcpServers",
        "collivind",
        {
            "type": "local",
            "command": _mcp_interpreter(),
            "args": ["-m", "collivind.mcp.server"],
            "tools": ["*"],
        },
    )


def _register_mcp_server():
    """Register Collivind as an MCP server with Claude Code, plus Codex when
    ~/.codex exists."""
    click.echo("Registering MCP server with Claude Code... ", nl=False)
    interpreter = _mcp_interpreter()
    # `-s user` so it applies everywhere. The old `--global` is not a flag the
    # CLI has ever accepted: it exited with "unknown option" and wrote nothing,
    # so this registration had never once succeeded — and capture_output hid
    # the reason behind a bare "registration failed".
    add = ["claude", "mcp", "add", "-s", "user", "collivind", "--", interpreter, "-m", "collivind.mcp.server"]

    known, problem = claude_mcp_servers()
    if known is None:
        click.secho(f"skipped ({problem})", fg="yellow")
        click.echo(f"Run manually: {shlex.join(add)}")
        return
    # Registering on top of an existing server is never what the user wants:
    # the plugin already provides one, `add` cannot update an entry in place,
    # and its collision check is per-scope, so a same-named entry in another
    # scope would silently keep winning. Leave whatever is there alone.
    if existing := _collivind_already_registered(set(known)):
        if _is_healthy(known[existing]):
            click.secho(f"already registered as {existing}", fg="green")
            return
        # Calling a dead entry "already registered" tells the person who
        # filed #33 — whose entry is exactly this — that nothing is wrong.
        click.secho(f"already registered as {existing}, but it is not answering", fg="yellow")
        click.echo(f"  {known[existing] or 'no status reported'}")
        click.echo(f"  Replace it with: claude mcp remove {existing} && collivind init")
        return

    try:
        subprocess.run(add, capture_output=True, check=True, text=True)
        click.secho("done", fg="green")
    except OSError:
        click.secho("skipped (claude CLI not found)", fg="yellow")
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip().splitlines()
        click.secho(f"skipped ({detail[0] if detail else 'registration failed'})", fg="yellow")
        click.echo(f"Run manually: {shlex.join(add)}")

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
        ("VS Code", register_vscode_profile_mcp, "your VS Code user mcp.json"),
        ("Copilot CLI", register_copilot_cli_mcp, "~/.copilot/mcp-config.json"),
    ):
        try:
            changed = registrar()
        except OSError as e:
            click.secho(f"Registering MCP server with {label}... skipped ({e})", fg="yellow")
            click.echo(f"Add the collivind entry to {hint} manually.")
            continue
        if changed:
            click.secho(f"Registering MCP server with {label}... done", fg="green")
