import click

from collivind.config import load_config
from collivind.engine.memory_manager import MemoryManager
from collivind.models import SearchQuery
from collivind.storage.factory import create_all_backends


@click.command()
@click.argument("query")
@click.option("--project", "-p", default="default", help="Project to search in")
@click.option("--category", "-c", default=None, help="Filter by category")
@click.option("--limit", "-l", default=10, help="Max results")
def search(query, project, category, limit):
    """Search memories from the terminal."""
    config = load_config()

    try:
        vector_store, graph_store, embedding_provider = create_all_backends(config)
        manager = MemoryManager(vector_store, graph_store, embedding_provider, config)

        q = SearchQuery(
            query=query,
            project_id=project,
            category=category,
            limit=limit,
        )
        results = manager.search(q)

        if not results:
            click.echo("No memories found.")
            return

        for r in results:
            cat = r.memory.category.value if hasattr(r.memory.category, "value") else r.memory.category
            score = f"{r.score:.2f}"
            click.secho(f"[{score}] ", fg="cyan", nl=False)
            click.secho(f"[{cat}] ", fg="yellow", nl=False)
            click.echo(r.memory.content)
            if r.related_entities:
                click.secho(f"       entities: {', '.join(r.related_entities)}", fg="bright_black")
            created = r.memory.created_at.strftime("%Y-%m-%d") if r.memory.created_at else "unknown"
            click.secho(f"       project: {r.memory.project_id} | {created}", fg="bright_black")
            click.echo()
    except Exception as e:
        click.secho(f"Search failed: {e}", fg="red")
