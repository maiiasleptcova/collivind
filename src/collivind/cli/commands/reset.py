import click

from collivind.config import load_config
from collivind.storage.factory import create_graph_store, create_vector_store


@click.command()
@click.confirmation_option(prompt="This will permanently delete ALL memories. Are you sure?")
def reset():
    """Reset all Collivind data (memories, entities, relationships)."""
    config = load_config()

    click.echo(f"Mode: {config.mode}")

    click.echo("Clearing vector store... ", nl=False)
    try:
        vector = create_vector_store(config)
        vector.delete_collection()
        vector.initialize()
        click.secho("done", fg="green")
    except Exception as e:
        click.secho(f"failed: {e}", fg="red")

    click.echo("Clearing graph store... ", nl=False)
    try:
        graph = create_graph_store(config)
        graph.initialize()
        graph.clear_all()
        click.secho("done", fg="green")
    except Exception as e:
        click.secho(f"failed: {e}", fg="red")

    click.secho("\nMemories reset successfully.", fg="green")
