import json
from unittest.mock import MagicMock, patch

from collivind.update_check import get_update_notice


def _pypi_response(version):
    resp = MagicMock()
    resp.json.return_value = {"info": {"version": version}}
    return resp


def test_notice_when_newer_version(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch("collivind.update_check.httpx.get", return_value=_pypi_response("99.0.0")):
        notice = get_update_notice()
    assert notice is not None
    assert "99.0.0" in notice
    assert "pip install -U collivind-memory" in notice


def test_no_notice_when_current(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    from collivind.version import __version__

    with patch("collivind.update_check.httpx.get", return_value=_pypi_response(__version__)):
        assert get_update_notice() is None


def test_check_throttled_to_daily(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch("collivind.update_check.httpx.get", return_value=_pypi_response("99.0.0")) as get:
        assert get_update_notice() is not None
        assert get_update_notice() is not None  # served from cache
        assert get.call_count == 1

    state = json.loads((tmp_path / ".collivind" / "update_check.json").read_text())
    assert state["latest"] == "99.0.0"


def test_silent_on_network_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch("collivind.update_check.httpx.get", side_effect=OSError("offline")):
        assert get_update_notice() is None


def test_cli_prints_notice_to_stderr(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from collivind.cli.main import cli

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch("collivind.update_check.get_update_notice", return_value="UPGRADE NOTICE"):
        result = CliRunner().invoke(cli, ["commands", "install", "--tool", "claude"])
    assert result.exit_code == 0
    assert "UPGRADE NOTICE" in result.stderr
