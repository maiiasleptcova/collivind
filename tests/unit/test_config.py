import pytest
from pathlib import Path
from collivind.config import load_config, CollivindConfig, DockerConfig

def test_default_config():
    # If no config.toml exists, it should load defaults
    config = load_config()
    assert isinstance(config, CollivindConfig)
    assert config.user_id == "local"
    assert config.docker.compose_project == "collivind"
    assert config.qdrant.port == 6333
    assert config.neo4j.user == "neo4j"
    assert config.expanded_data_dir == Path("~/.collivind").expanduser()
