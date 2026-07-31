import webbrowser

import click


@click.command()
@click.option("--port", default=8765, show_default=True, help="Port to bind on localhost")
@click.option("--open-browser/--no-open-browser", default=True, help="Open the UI on start")
def web(port, open_browser):
    """Browse, search and edit memories in a local web UI."""
    from collivind.config import load_config
    from collivind.web.api import MemoryAPI
    from collivind.web.server import build_server

    config = load_config()
    try:
        from collivind.cli.commands.memory import _manager

        manager = _manager()
    except Exception as e:
        raise click.ClickException(f"{e}\n\nThe UI reads the same store as the hooks; fix the mode first.")

    server = build_server(MemoryAPI(manager, mode=config.mode), port=port)
    url = f"http://127.0.0.1:{port}"
    click.secho(f"Collivind UI on {url}", fg="green")
    click.echo(f"Reading the {config.mode} store. Ctrl-C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        server.server_close()
