import subprocess
import time
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

    click.echo("Initializing embedded Qdrant... ", nl=False)
    try:
        from collivind.storage.qdrant_embedded import EmbeddedQdrantStore
        qdrant = EmbeddedQdrantStore(
            data_dir=str(data_dir),
            config=config.qdrant,
            dimension=config.embeddings.dimension
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


def _init_docker(config, data_dir: Path):
    """Initialize Docker mode: start containers."""
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


def _register_mcp_server():
    """Register Collivind as an MCP server with Claude Code."""
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
