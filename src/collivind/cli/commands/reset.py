import click

from collivind.config import load_config
from collivind.storage.qdrant_store import QdrantVectorStore
from collivind.storage.neo4j_store import Neo4jGraphStore


@click.command()
@click.confirmation_option(prompt="This will permanently delete ALL memories. Are you sure?")
def reset():
    """Reset all Collivind data (memories, entities, relationships)."""
    config = load_config()

    click.echo("Clearing Qdrant collection... ", nl=False)
    try:
        qdrant = QdrantVectorStore(config.qdrant, config.embeddings.dimension)
        qdrant.delete_collection()
        qdrant.initialize()
        click.secho("done", fg="green")
    except Exception as e:
        click.secho(f"failed: {e}", fg="red")

    click.echo("Clearing Neo4j database... ", nl=False)
    try:
        neo4j = Neo4jGraphStore(config.neo4j)
        neo4j.clear_all()
        neo4j.initialize()
        click.secho("done", fg="green")
    except Exception as e:
        click.secho(f"failed: {e}", fg="red")

    click.secho("\nMemories reset successfully.", fg="green")
