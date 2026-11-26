"""`collivind describe`: show or store what a topic is, in one or two sentences.

Collivind never writes the text; this is the terminal twin of the
`collivind_describe_topic` MCP tool. The memory wiki opens the topic's page with it.
"""

import sys

import click

from collivind.engine.descriptions import DESCRIPTION_KEY, describe_entity


@click.command()
@click.argument("name")
@click.argument("text", required=False)
def describe(name, text):
    """Show NAME's description, or store TEXT as it ("-" reads stdin)."""
    from collivind.cli.commands.memory import _fail, _manager

    try:
        manager = _manager()
        if text is None:
            entity = manager.graph_store.get_entity(manager.canonical_entity_name(name))
            if entity is None:
                raise LookupError(f"No entity named {name!r}")
            current = (entity.properties or {}).get(DESCRIPTION_KEY)
            click.echo(current or f"{entity.name} has no description yet.")
            return
        result = describe_entity(manager, name, sys.stdin.read() if text == "-" else text)
    except (LookupError, ValueError) as e:
        _fail(f"Error: {e.args[0]}")
        return
    click.secho(f"Described {result['entity']}.", fg="green")
