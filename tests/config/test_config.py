"""Tests for config module."""

import os

from decimal import Decimal

import pytest

from app.config import Settings, get_config, reset_config
from app.config.settings import (
    _apply_env_overrides,
    _cast_value,
    _dict_to_dataclass,
    validate_settings,
)


class TestCastValue:
    def test_bool_true(self):
        assert _cast_value("true") is True
        assert _cast_value("True") is True

    def test_bool_false(self):
        assert _cast_value("false") is False
        assert _cast_value("False") is False

    def test_int(self):
        assert _cast_value("42") == 42

    def test_float(self):
        assert _cast_value("0.00025") == pytest.approx(0.00025)

    def test_string(self):
        assert _cast_value("hello") == "hello"


class TestDictToDataclass:
    def test_full_mapping(self):
        from app.config.settings import ProjectConfig

        data = {"name": "test", "env": "prod", "debug": True}
        cfg = _dict_to_dataclass(ProjectConfig, data)
        assert cfg.name == "test"
        assert cfg.env == "prod"
        assert cfg.debug is True

    def test_partial_mapping(self):
        from app.config.settings import ProjectConfig

        data = {"name": "test"}
        cfg = _dict_to_dataclass(ProjectConfig, data)
        assert cfg.name == "test"
        assert cfg.env == "dev"
        assert cfg.debug is False

    def test_ignores_unknown_keys(self):
        from app.config.settings import ProjectConfig

        data = {"name": "test", "unknown_key": "value"}
        cfg = _dict_to_dataclass(ProjectConfig, data)
        assert cfg.name == "test"
        assert not hasattr(cfg, "unknown_key")


class TestApplyEnvOverrides:
    def test_override_existing(self):
        raw = {"data": {"db_path": "./default"}}
        os.environ["PLUM_DATA_DB_PATH"] = "/custom/path"
        result = _apply_env_overrides(raw)
        assert result["data"]["db_path"] == "/custom/path"
        del os.environ["PLUM_DATA_DB_PATH"]

    def test_no_override_if_no_env(self):
        raw = {"data": {"db_path": "./default"}}
        result = _apply_env_overrides(raw)
        assert result["data"]["db_path"] == "./default"

    def test_bool_override(self):
        raw = {"project": {"debug": False}}
        os.environ["PLUM_PROJECT_DEBUG"] = "true"
        result = _apply_env_overrides(raw)
        assert result["project"]["debug"] is True
        del os.environ["PLUM_PROJECT_DEBUG"]


class TestValidateSettings:
    def test_valid_settings(self):
        settings = Settings()
        validate_settings(settings)

    def test_empty_db_path(self):
        settings = Settings(data=Settings().data)
        settings.data.db_path = ""
        with pytest.raises(ValueError, match="data.db_path"):
            validate_settings(settings)

    def test_negative_capital(self):
        settings = Settings()
        settings.engine.initial_capital = -100
        with pytest.raises(ValueError, match="engine.initial_capital"):
            validate_settings(settings)

    def test_invalid_commission(self):
        settings = Settings()
        settings.cost.commission_rate = 1.5
        with pytest.raises(ValueError, match="cost.commission_rate"):
            validate_settings(settings)

    def test_invalid_max_position(self):
        settings = Settings()
        settings.risk.max_position_pct = 1.5
        with pytest.raises(ValueError, match="risk.max_position_pct"):
            validate_settings(settings)


class TestGetConfig:
    def test_returns_singleton(self):
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_returns_settings_instance(self):
        reset_config()
        config = get_config()
        assert isinstance(config, Settings)

    def test_default_values(self):
        reset_config()
        config = get_config()
        assert config.project.name == "plum"
        assert config.engine.initial_capital == 1_000_000
        assert config.cost.commission_rate == Decimal("0.00025")
