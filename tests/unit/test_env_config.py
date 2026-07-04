import os
from unittest.mock import patch

from collivind.config import CollivindConfig, _apply_env_overrides


def test_shared_api_key_maps_to_both():
    config = CollivindConfig()
    with patch.dict(os.environ, {"COLLIVIND_API_KEY": "sk-shared-123"}, clear=False):
        _apply_env_overrides(config)
    assert config.embeddings.api_key == "sk-shared-123"
    assert config.qdrant.api_key == "sk-shared-123"


def test_per_service_key_overrides_shared():
    config = CollivindConfig()
    env = {
        "COLLIVIND_API_KEY": "sk-shared",
        "COLLIVIND_EMBEDDINGS_API_KEY": "sk-emb-specific",
        "COLLIVIND_QDRANT_API_KEY": "qd-specific",
    }
    with patch.dict(os.environ, env, clear=False):
        _apply_env_overrides(config)
    assert config.embeddings.api_key == "sk-emb-specific"
    assert config.qdrant.api_key == "qd-specific"


def test_shared_key_does_not_overwrite_config_file_value():
    config = CollivindConfig()
    config.embeddings.api_key = "from-config-file"
    with patch.dict(os.environ, {"COLLIVIND_API_KEY": "sk-shared"}, clear=False):
        _apply_env_overrides(config)
    assert config.embeddings.api_key == "from-config-file"


def test_per_service_key_overwrites_config_file_value():
    config = CollivindConfig()
    config.embeddings.api_key = "from-config-file"
    with patch.dict(os.environ, {"COLLIVIND_EMBEDDINGS_API_KEY": "sk-override"}, clear=False):
        _apply_env_overrides(config)
    assert config.embeddings.api_key == "sk-override"


def test_qdrant_url_env():
    config = CollivindConfig()
    with patch.dict(os.environ, {"COLLIVIND_QDRANT_URL": "https://cloud.qdrant.io"}, clear=False):
        _apply_env_overrides(config)
    assert config.qdrant.url == "https://cloud.qdrant.io"


def test_mode_env():
    config = CollivindConfig()
    with patch.dict(os.environ, {"COLLIVIND_MODE": "remote"}, clear=False):
        _apply_env_overrides(config)
    assert config.mode == "remote"


def test_no_env_vars_leaves_defaults():
    config = CollivindConfig()
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("COLLIVIND_")}
    with patch.dict(os.environ, clean_env, clear=True):
        _apply_env_overrides(config)
    assert config.embeddings.api_key == ""
    assert config.qdrant.api_key == ""
