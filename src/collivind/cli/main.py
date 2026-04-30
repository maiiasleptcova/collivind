import click
from collivind.version import __version__
from collivind.cli.commands.init import init
from collivind.cli.commands.status import status
from collivind.cli.commands.hook import hook

@click.group()
@click.version_option(version=__version__)
def cli():
    """Collivind - Graph-based memory layer for AI coding assistants."""
    pass

cli.add_command(init)
cli.add_command(status)
cli.add_command(hook)
