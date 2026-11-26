"""The public site's install commands pin a release wheel; keep them on the current version."""

from pathlib import Path

from collivind.version import __version__


def test_site_installs_current_release():
    html = (Path(__file__).parents[2] / "site" / "index.html").read_text()
    wheel = f"releases/download/v{__version__}/collivind_memory-{__version__}-py3-none-any.whl"
    assert html.count(wheel) == 2, f"site/index.html must install v{__version__}: update both install commands"
    assert html.count("releases/download/") == 2
