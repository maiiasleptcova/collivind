import click

from collivind.cli.commands.agent_commands import commands
from collivind.cli.commands.docker import docker
from collivind.cli.commands.hook import hook
from collivind.cli.commands.init import init
from collivind.cli.commands.memory import (
    add,
    context,
    export_cmd,
    forget,
    get,
    import_cmd,
    invalidate,
    sync,
    update,
)
from collivind.cli.commands.reset import reset
from collivind.cli.commands.search import search
from collivind.cli.commands.status import status
from collivind.version import __version__


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """Collivind - Graph-based memory layer for AI coding assistants."""
    # hooks run on every prompt/response — no update checks on that path
    if ctx.invoked_subcommand == "hook":
        return
    try:
        from collivind.config import load_config
        from collivind.update_check import get_update_notice

        if load_config().check_updates:
            notice = get_update_notice()
            if notice:
                click.secho(notice, fg="yellow", err=True)
    except Exception:
        pass


cli.add_command(init)
cli.add_command(status)
cli.add_command(hook)
cli.add_command(search)
cli.add_command(reset)
cli.add_command(docker)
cli.add_command(add)
cli.add_command(get)
cli.add_command(context)
cli.add_command(update)
cli.add_command(invalidate)
cli.add_command(forget)
cli.add_command(export_cmd)
cli.add_command(import_cmd)
cli.add_command(sync)
cli.add_command(commands)
