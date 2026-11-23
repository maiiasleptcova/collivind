import webbrowser

import click


@click.command()
@click.option("--port", default=8765, show_default=True, help="Port to bind")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to bind. Keep loopback; the container uses 0.0.0.0 behind a 127.0.0.1 port mapping.",
)
@click.option("--open-browser/--no-open-browser", default=True, help="Open the UI on start")
def web(port, host, open_browser):
    """Explore, search and edit memories in a local web UI."""
    from collivind.config import load_config
    from collivind.web.api import MemoryAPI
    from collivind.web.container import embedded_refusal, in_container
    from collivind.web.server import build_server

    config = load_config()
    if in_container() and config.mode == "embedded":
        reason = embedded_refusal(config.expanded_data_dir)
        if reason:
            raise click.ClickException(f"Refusing to open the embedded store: {reason}")
    try:
        from collivind.cli.commands.memory import _manager

        manager = _manager()
    except Exception as e:
        raise click.ClickException(f"{e}\n\nThe UI reads the same store as the hooks; fix the mode first.")

    server = build_server(MemoryAPI(manager, mode=config.mode), host=host, port=port)
    url = f"http://127.0.0.1:{port}"
    click.secho(f"Collivind UI on {url}", fg="green")
    if host not in ("127.0.0.1", "localhost", "::1"):
        click.echo(f"Bound to {host}; requests must still address it as localhost.")
    click.echo(f"Reading the {config.mode} store. Ctrl-C to stop.")
    if open_browser and not in_container():
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        server.server_close()
