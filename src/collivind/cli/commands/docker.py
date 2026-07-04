import subprocess

import click

from collivind.config import load_config
from collivind.docker.compose import docker_compose_down, docker_compose_up


@click.group()
def docker():
    """Manage Collivind Docker containers."""
    pass


@docker.command()
def up():
    """Start Collivind Docker containers."""
    config = load_config()
    data_dir = config.expanded_data_dir
    click.echo("Starting Collivind containers...")
    try:
        docker_compose_up(data_dir)
        click.secho("Containers started.", fg="green")
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
