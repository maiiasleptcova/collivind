import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from collivind.cli.commands.docker import docker
from collivind.cli.commands.init import init
from collivind.cli.commands.reset import reset
from collivind.cli.commands.search import search
from collivind.cli.commands.status import status
from collivind.config import CollivindConfig


def _mock_config(mode="docker"):
    return CollivindConfig(mode=mode, data_dir=tempfile.mkdtemp())


class TestInitCommand:
    @patch("collivind.cli.commands.init._register_mcp_server")
    @patch("collivind.cli.commands.init.check_all_services")
    @patch("collivind.cli.commands.init.docker_compose_up")
    @patch("collivind.cli.commands.init.copy_templates")
    @patch("collivind.cli.commands.init.check_docker_running")
    @patch("collivind.cli.commands.init.load_config")
    def test_init_docker_mode(self, mock_config, mock_docker, mock_templates, mock_up, mock_health, mock_mcp):
        mock_config.return_value = _mock_config("docker")
        mock_health.return_value = {
            "qdrant": {"status": "ok", "message": "healthy"},
            "neo4j": {"status": "ok", "message": "healthy"},
            "embeddings": {"status": "ok", "message": "healthy"},
        }
        runner = CliRunner()
        result = runner.invoke(init)
        assert result.exit_code == 0
        assert "Collivind is ready" in result.output
        mock_docker.assert_called_once()
        mock_templates.assert_called_once()
        mock_up.assert_called_once()

    @patch("collivind.cli.commands.init._register_mcp_server")
    @patch("collivind.cli.commands.init.load_config")
    def test_init_embedded_mode(self, mock_config, mock_mcp):
        config = _mock_config("embedded")
        mock_config.return_value = config
        runner = CliRunner()
        with patch("collivind.storage.graph_sqlite.SqliteGraphStore") as mock_graph, \
             patch("collivind.storage.qdrant_embedded.EmbeddedQdrantStore") as mock_qdrant:
            mock_graph.return_value = MagicMock()
            mock_qdrant.return_value = MagicMock()
            result = runner.invoke(init)
        assert result.exit_code == 0
        assert "embedded" in result.output.lower()

    @patch("collivind.cli.commands.init._register_mcp_server")
    @patch("collivind.cli.commands.init.load_config")
    def test_init_generates_config_toml(self, mock_config, mock_mcp):
        config = _mock_config("embedded")
        mock_config.return_value = config
        runner = CliRunner()
        with patch("collivind.storage.graph_sqlite.SqliteGraphStore") as mock_graph, \
             patch("collivind.storage.qdrant_embedded.EmbeddedQdrantStore") as mock_qdrant:
            mock_graph.return_value = MagicMock()
            mock_qdrant.return_value = MagicMock()
            runner.invoke(init)
        config_path = Path(config.data_dir) / "config.toml"
        assert config_path.exists()
        content = config_path.read_text()
        assert 'mode = "embedded"' in content


class TestStatusCommand:
    @patch("collivind.cli.commands.status.load_config")
    @patch("collivind.cli.commands.status.check_all_services")
    def test_status_docker(self, mock_health, mock_config):
        mock_config.return_value = _mock_config("docker")
        mock_health.return_value = {
            "qdrant": {"status": "ok", "message": "Qdrant is healthy"},
            "neo4j": {"status": "error", "message": "Connection failed"},
            "embeddings": {"status": "ok", "message": "Embeddings ok"},
        }
        runner = CliRunner()
        result = runner.invoke(status)
        assert result.exit_code == 0
        assert "Mode: docker" in result.output
        assert "qdrant" in result.output
        assert "neo4j" in result.output

    @patch("collivind.storage.factory.create_all_backends")
    @patch("collivind.cli.commands.status.load_config")
    def test_status_embedded(self, mock_config, mock_backends):
        mock_config.return_value = _mock_config("embedded")
        v, g, e = MagicMock(), MagicMock(), MagicMock()
        v.health_check.return_value = {"status": "ok", "message": "Embedded Qdrant"}
        g.health_check.return_value = {"status": "ok", "message": "SQLite graph"}
        e.health_check.return_value = {"status": "ok", "message": "Local model"}
        mock_backends.return_value = (v, g, e)
        runner = CliRunner()
        result = runner.invoke(status)
        assert result.exit_code == 0
        assert "Mode: embedded" in result.output


class TestSearchCommand:
    @patch("collivind.cli.commands.search.create_all_backends")
    @patch("collivind.cli.commands.search.load_config")
    def test_search_no_results(self, mock_config, mock_backends):
        mock_config.return_value = _mock_config("docker")
        v, g, e = MagicMock(), MagicMock(), MagicMock()
        e.embed.return_value = [0.0] * 384
        v.search.return_value = []
        mock_backends.return_value = (v, g, e)
        runner = CliRunner()
        result = runner.invoke(search, ["test query"])
        assert result.exit_code == 0
        assert "No memories found" in result.output

    @patch("collivind.cli.commands.search.create_all_backends")
    @patch("collivind.cli.commands.search.load_config")
    def test_search_error(self, mock_config, mock_backends):
        mock_config.return_value = _mock_config("docker")
        mock_backends.side_effect = Exception("connection refused")
        runner = CliRunner()
        result = runner.invoke(search, ["test query"])
        assert result.exit_code == 0
        assert "Search failed" in result.output


class TestResetCommand:
    @patch("collivind.cli.commands.reset.create_graph_store")
    @patch("collivind.cli.commands.reset.create_vector_store")
    @patch("collivind.cli.commands.reset.load_config")
    def test_reset_confirms(self, mock_config, mock_vector, mock_graph):
        mock_config.return_value = _mock_config("docker")
        mock_vector.return_value = MagicMock()
        mock_graph.return_value = MagicMock()
        runner = CliRunner()
        result = runner.invoke(reset, input="y\n")
        assert result.exit_code == 0
        assert "reset successfully" in result.output

    @patch("collivind.cli.commands.reset.load_config")
    def test_reset_aborted(self, mock_config):
        mock_config.return_value = _mock_config("docker")
        runner = CliRunner()
        result = runner.invoke(reset, input="n\n")
        assert result.exit_code != 0


class TestDockerCommand:
    @patch("collivind.cli.commands.docker.docker_compose_up")
    @patch("collivind.cli.commands.docker.load_config")
    def test_docker_up(self, mock_config, mock_up):
        mock_config.return_value = _mock_config("docker")
        runner = CliRunner()
        result = runner.invoke(docker, ["up"])
        assert result.exit_code == 0
        assert "started" in result.output.lower()

    @patch("collivind.cli.commands.docker.docker_compose_down")
    @patch("collivind.cli.commands.docker.load_config")
    def test_docker_down(self, mock_config, mock_down):
        mock_config.return_value = _mock_config("docker")
        runner = CliRunner()
        result = runner.invoke(docker, ["down"])
        assert result.exit_code == 0
        assert "stopped" in result.output.lower()
