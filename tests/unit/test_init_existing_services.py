"""Issue #17: collivind may run inside a container that already has Qdrant and
Neo4j. Provisioning containers from there is both impossible and unnecessary."""

from unittest.mock import patch

import click
from click.testing import CliRunner

from collivind.config import CollivindConfig

HEALTHY = {
    "qdrant": {"status": "ok", "message": "up"},
    "neo4j": {"status": "ok", "message": "up"},
    "embeddings": {"status": "ok", "message": "up"},
}
DOWN = {
    "qdrant": {"status": "error", "message": "connection refused"},
    "neo4j": {"status": "ok", "message": "up"},
    "embeddings": {"status": "ok", "message": "up"},
}


def _run(config):
    from collivind.cli.commands.init import _init_docker

    @click.command()
    def probe():
        _init_docker(config, config.expanded_data_dir)

    return CliRunner().invoke(probe)


def test_skips_provisioning_when_services_already_healthy(tmp_path):
    """The in-container case: no daemon, nothing to start, services present."""
    config = CollivindConfig(mode="docker", data_dir=str(tmp_path))
    with (
        patch("collivind.cli.commands.init.check_all_services", return_value=HEALTHY),
        patch("collivind.cli.commands.init.check_docker_running") as daemon,
        patch("collivind.cli.commands.init.docker_compose_up") as up,
        patch("collivind.cli.commands.init.copy_templates") as templates,
    ):
        result = _run(config)

    assert result.exit_code == 0
    daemon.assert_not_called(), "must not require a Docker daemon when services are up"
    up.assert_not_called(), "must not start containers that already run"
    templates.assert_not_called()
    assert "already-running" in result.output.lower() or "already running" in result.output.lower()
    assert "Collivind is ready" in result.output, "the success banner must not be skipped"


def test_provisions_when_services_are_down(tmp_path):
    """Normal laptop case is unchanged."""
    config = CollivindConfig(mode="docker", data_dir=str(tmp_path))
    with (
        patch("collivind.cli.commands.init.check_all_services", side_effect=[DOWN, HEALTHY]),
        patch("collivind.cli.commands.init.check_docker_running") as daemon,
        patch("collivind.cli.commands.init.docker_compose_up") as up,
        patch("collivind.cli.commands.init.copy_templates") as templates,
    ):
        result = _run(config)

    assert result.exit_code == 0
    daemon.assert_called_once()
    up.assert_called_once()
    templates.assert_called_once()


def test_partial_health_still_provisions(tmp_path):
    """One backend down means the stack is not usable — provision, don't assume."""
    config = CollivindConfig(mode="docker", data_dir=str(tmp_path))
    with (
        patch("collivind.cli.commands.init.check_all_services", side_effect=[DOWN, HEALTHY]),
        patch("collivind.cli.commands.init.check_docker_running"),
        patch("collivind.cli.commands.init.docker_compose_up") as up,
        patch("collivind.cli.commands.init.copy_templates"),
    ):
        _run(config)
    up.assert_called_once()


def test_probe_failure_is_treated_as_down_not_fatal(tmp_path):
    """A probe that raises must fall through to provisioning, not crash init."""
    config = CollivindConfig(mode="docker", data_dir=str(tmp_path))
    with (
        patch("collivind.cli.commands.init.check_all_services", side_effect=[OSError("no route"), HEALTHY]),
        patch("collivind.cli.commands.init.check_docker_running"),
        patch("collivind.cli.commands.init.docker_compose_up") as up,
        patch("collivind.cli.commands.init.copy_templates"),
    ):
        result = _run(config)
    assert result.exit_code == 0
    up.assert_called_once()
