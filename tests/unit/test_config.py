import tempfile
from pathlib import Path

from collivind.config import CollivindConfig, generate_default_config, load_config


def test_default_config():
    config = load_config()
    assert isinstance(config, CollivindConfig)
    assert config.user_id == "local"
    assert config.docker.compose_project == "collivind"
    assert config.qdrant.port == 6333
    assert config.neo4j.user == "neo4j"
    assert config.expanded_data_dir == Path("~/.collivind").expanduser()

def test_default_config_mode():
    config = load_config()
    assert config.mode == "docker"

def test_generate_default_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.toml"
        generate_default_config(path, mode="embedded")
        assert path.exists()
        content = path.read_text()
        assert 'mode = "embedded"' in content
        assert "[search]" in content
        assert "[hooks]" in content
