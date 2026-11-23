import subprocess

import click

from collivind.config import load_config
from collivind.docker.compose import WEB_PORT, copy_templates, docker_compose_down, docker_compose_up


@click.group()
def docker():
    """Manage Collivind Docker containers."""
    pass


@docker.command()
def up():
    """Start Collivind Docker containers, refreshing the templates first.

    init writes the templates only once, so without the refresh an existing
    install never received fixes like the 0.10.0 healthchecks (#39).
    """
    config = load_config()
    data_dir = config.expanded_data_dir
    click.echo("Starting Collivind containers...")
    try:
        refreshed = copy_templates(data_dir, config)
        if refreshed == "replaced":
            click.echo(f"Updated docker-compose.yml; previous kept as {data_dir / 'docker-compose.yml.bak'}")
        elif refreshed == "kept-edits":
            click.secho(
                f"{data_dir / 'docker-compose.yml'} has local edits, so it was not updated to the current "
                "template. Delete it and re-run `collivind docker up` to take the new one.",
                fg="yellow",
            )
        docker_compose_up(data_dir)
        click.secho("Containers started.", fg="green")
        if refreshed != "kept-edits":
            click.echo(f"Memory explorer: http://127.0.0.1:{WEB_PORT} (loopback only)")
    except Exception as e:
        click.secho(f"Failed: {e}", fg="red")


@docker.command()
def down():
    """Stop Collivind Docker containers."""
    config = load_config()
    data_dir = config.expanded_data_dir
    click.echo("Stopping Collivind containers...")
    try:
        docker_compose_down(data_dir)
        click.secho("Containers stopped.", fg="green")
    except Exception as e:
        click.secho(f"Failed: {e}", fg="red")


@docker.command()
def logs():
    """Show Docker container logs."""
    config = load_config()
    data_dir = config.expanded_data_dir
    try:
        subprocess.run(
            ["docker-compose", "logs", "--tail=50"],
            cwd=str(data_dir),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.secho(f"Failed to get logs: {e}", fg="red")
    except FileNotFoundError:
        click.secho("docker-compose not found.", fg="red")
