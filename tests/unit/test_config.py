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


def test_default_template_matches_search_dataclass():
    """Guard against template/dataclass drift: every SearchConfig field must
    appear in the template with its default value."""
    import dataclasses
    import tomllib

    from collivind.config import DEFAULT_CONFIG_TEMPLATE, HooksConfig, SearchConfig

    data = tomllib.loads(DEFAULT_CONFIG_TEMPLATE.format(mode="docker"))

    for cls, section in ((SearchConfig, "search"), (HooksConfig, "hooks")):
        for f in dataclasses.fields(cls):
            assert f.name in data[section], f"[{section}] missing {f.name}"
            assert data[section][f.name] == f.default
