from unittest.mock import MagicMock, patch

from collivind.config import CollivindConfig
from collivind.docker.health import check_neo4j_health


def test_neo4j_health_uses_configured_uri():
    config = CollivindConfig()
    config.neo4j.uri = "bolt://db.example.com:9999"

    with patch("neo4j.GraphDatabase") as gdb:
        gdb.driver.return_value = MagicMock()
        result = check_neo4j_health(config)

    assert result["status"] == "ok"
    assert gdb.driver.call_args.args == ("bolt://db.example.com:9999",)


def test_neo4j_health_reports_connection_failure():
    config = CollivindConfig()

    with patch("neo4j.GraphDatabase") as gdb:
        gdb.driver.return_value.verify_connectivity.side_effect = OSError("refused")
        result = check_neo4j_health(config)

    assert result["status"] == "error"
    assert "refused" in result["message"]
