"""Existing installs must receive template fixes (#39).

`collivind init` writes the compose files once, and `collivind docker up`
used to start whatever was already in the data dir, so the 0.10.0
healthchecks and named /logs volume never reached anyone already installed.
"""

from unittest.mock import patch

from click.testing import CliRunner

from collivind.cli.commands.docker import docker
from collivind.config import CollivindConfig
from collivind.docker.compose import copy_templates

OLD_COMPOSE = """services:
  neo4j:
    image: neo4j:5.26-community
    environment:
      - NEO4J_AUTH=neo4j/s3cret-generated
    volumes:
      - neo4j-data:/data
"""


def _config(tmp_path, password="from-config"):
    config = CollivindConfig()
    config.data_dir = str(tmp_path)
    config.neo4j.password = password
    return config


def test_refresh_keeps_the_password_the_database_was_created_with(tmp_path):
    """NEO4J_AUTH only applies on a fresh volume, so changing it in compose
    would do nothing but desynchronise the file from the running database."""
    (tmp_path / "docker-compose.yml").write_text(OLD_COMPOSE)

    copy_templates(tmp_path, _config(tmp_path))

    text = (tmp_path / "docker-compose.yml").read_text()
    assert "NEO4J_AUTH=neo4j/s3cret-generated" in text
    assert "from-config" not in text
    assert "healthcheck" in text and "neo4j-logs:/logs" in text


def test_refresh_backs_up_a_compose_file_it_changes(tmp_path):
    (tmp_path / "docker-compose.yml").write_text(OLD_COMPOSE)

    copy_templates(tmp_path, _config(tmp_path))

    assert (tmp_path / "docker-compose.yml.bak").read_text() == OLD_COMPOSE


def test_refresh_of_an_up_to_date_file_leaves_no_backup(tmp_path):
    config = _config(tmp_path)
    copy_templates(tmp_path, config)
    copy_templates(tmp_path, config)

    assert not (tmp_path / "docker-compose.yml.bak").exists()


def test_fresh_install_uses_the_config_password(tmp_path):
    copy_templates(tmp_path, _config(tmp_path))

    assert "NEO4J_AUTH=neo4j/from-config" in (tmp_path / "docker-compose.yml").read_text()


def test_docker_up_refreshes_templates_before_starting(tmp_path):
    (tmp_path / "docker-compose.yml").write_text(OLD_COMPOSE)
    order = []

    with (
        patch("collivind.cli.commands.docker.load_config", return_value=_config(tmp_path)),
        patch(
            "collivind.cli.commands.docker.docker_compose_up",
            side_effect=lambda d: order.append(("up", (d / "docker-compose.yml").read_text())),
        ),
    ):
        result = CliRunner().invoke(docker, ["up"])

    assert result.exit_code == 0, result.output
    assert "docker-compose.yml.bak" in result.output
    assert len(order) == 1 and "healthcheck" in order[0][1]
    assert "s3cret-generated" in order[0][1]
