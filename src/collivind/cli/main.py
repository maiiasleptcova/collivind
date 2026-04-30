import click
from collivind.version import __version__
from collivind.cli.commands.init import init
from collivind.cli.commands.status import status
from collivind.cli.commands.hook import hook
from collivind.cli.commands.search import search
from collivind.cli.commands.reset import reset
from collivind.cli.commands.docker import docker

@click.group()
@click.version_option(version=__version__)
def cli():
    """Collivind - Graph-based memory layer for AI coding assistants."""
    pass

cli.add_command(init)
cli.add_command(status)
cli.add_command(hook)
cli.add_command(search)
cli.add_command(reset)
cli.add_command(docker)
