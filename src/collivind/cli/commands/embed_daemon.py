import time

import click

from collivind.storage import embed_daemon as daemon


def _socket():
    from collivind.config import load_config

    config = load_config()
    return config, daemon.socket_path_for(config.expanded_data_dir)


def _info(path):
    try:
        return daemon.request(path, {"op": "info"}, timeout=2.0)
    except Exception:
        return None


@click.group("embed-daemon")
def embed_daemon():
    """Resident embedding model for embedded mode (starts on demand)."""


@embed_daemon.command()
@click.option("--wait", default=120, show_default=True, help="Seconds to wait for the model to load")
def start(wait):
    """Start the daemon in the background and wait until it answers."""
    config, path = _socket()
    if _info(path):
        click.echo(f"Already running on {path}")
        return
    daemon.spawn(path, config.embeddings.model, config.embeddings.daemon_idle_timeout)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        info = _info(path)
        if info:
            click.secho(f"Running (pid {info['pid']}, {info['model']}) on {path}", fg="green")
            return
        time.sleep(0.5)
    raise click.ClickException(f"Daemon did not come up; see {path.with_name(daemon.LOG_NAME)}")


@embed_daemon.command()
def stop():
    """Stop the daemon."""
    _, path = _socket()
    try:
        daemon.request(path, {"op": "shutdown"}, timeout=5.0)
    except Exception:
        click.echo("Not running.")
        return
    click.echo("Stopped.")


@embed_daemon.command()
def status():
    """Show whether the daemon is running."""
    _, path = _socket()
    info = _info(path)
    if not info:
        if len(str(path)) > daemon.MAX_SOCKET_PATH:
            click.echo(
                f"Disabled: {path} is over the unix socket limit and the private fallback "
                f"under {daemon.SHORT_SOCKET_BASE} is unusable (see the log for why)."
            )
        else:
            click.echo(f"Not running (socket {path}).")
        return
    click.echo(f"Running: pid {info['pid']}, collivind {info['version']}, model {info['model']}, socket {path}")
