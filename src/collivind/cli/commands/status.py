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
