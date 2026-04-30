import click
import time
from pathlib import Path

from collivind.config import load_config
from collivind.docker.compose import check_docker_running, copy_templates, docker_compose_up
from collivind.docker.health import check_all_services

@click.command()
def init():
    """Initialize Collivind: setup Docker config and start containers."""
    click.echo("Initializing Collivind...")
    
    # Check if docker is running
    try:
        check_docker_running()
        click.echo("✓ Docker is running")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        return

    config = load_config()
    data_dir = config.expanded_data_dir
    
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
