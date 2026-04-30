import click
from collivind.config import load_config
from collivind.docker.health import check_all_services

@click.command()
def status():
    """Check health status of Collivind services."""
    config = load_config()
    click.echo("Checking Collivind services status...")
    
    status = check_all_services(config)
    
    for service, info in status.items():
        if info["status"] == "ok":
            click.secho(f"✓ {service}: {info['message']}", fg="green")
        else:
            click.secho(f"✗ {service}: {info['message']}", fg="red")
