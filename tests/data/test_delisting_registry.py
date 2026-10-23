"""Tests for delisting registry."""

from datetime import datetime

import pytest

from app.data.delisting_registry import (
    DelistingRegistry,
)


class TestDelistingRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return DelistingRegistry(tmp_path / "delisting_registry.json")

    def test_record_last_trading_day(self, registry):
        registry.record_last_trading_day(
            "000001.SZ", datetime(2024, 1, 1), reason="已退市"
        )
        assert registry.is_delisted("000001.SZ", datetime(2024, 1, 1)) is True
        assert registry.is_delisted("000001.SZ", datetime(2023, 12, 31)) is False

    def test_record_first_trading_day(self, registry):
        registry.record_first_trading_day("000001.SZ", datetime(2020, 1, 1))
        lifecycle = registry.get_lifecycle("000001.SZ")
        assert lifecycle is not None
        assert lifecycle.first_trading_day == datetime(2020, 1, 1)
        assert lifecycle.is_active is True

    def test_is_delisted_unknown_symbol(self, registry):
        assert registry.is_delisted("UNKNOWN", datetime(2024, 1, 1)) is False

    def test_can_trade_before_listing(self, registry):
        registry.record_first_trading_day("000001.SZ", datetime(2024, 1, 1))
        assert registry.can_trade("000001.SZ", datetime(2023, 12, 31)) is False
        assert registry.can_trade("000001.SZ", datetime(2024, 1, 1)) is True

    def test_can_trade_after_delisting(self, registry):
        registry.record_last_trading_day(
            "000001.SZ", datetime(2024, 1, 1), reason="已退市"
        )
        assert registry.can_trade("000001.SZ", datetime(2024, 1, 2)) is False

    def test_get_lifecycle_unknown_symbol(self, registry):
        assert registry.get_lifecycle("UNKNOWN") is None

    def test_get_all_delisted(self, registry):
        registry.record_last_trading_day(
            "000001.SZ", datetime(2024, 1, 1), reason="已退市"
        )
        registry.record_last_trading_day(
            "600000.SH", datetime(2024, 2, 1), reason="已退市"
        )
        delisted = registry.get_all_delisted()
        assert len(delisted) == 2
        assert "000001.SZ" in delisted
        assert "600000.SH" in delisted

    def test_get_all_active(self, registry):
        registry.record_first_trading_day("000001.SZ", datetime(2024, 1, 1))
        active = registry.get_all_active()
        assert len(active) == 1
        assert "000001.SZ" in active

    def test_get_delisting_events(self, registry):
        registry.record_last_trading_day(
            "000001.SZ", datetime(2024, 1, 1), reason="已退市"
        )
        registry.record_last_trading_day(
            "600000.SH", datetime(2024, 2, 1), reason="已退市"
        )
        events = registry.get_delisting_events()
        assert len(events) == 2
        assert events[0].symbol == "000001.SZ"
        assert events[1].symbol == "600000.SH"

    def test_merge_registries(self, registry, tmp_path):
        other = DelistingRegistry(tmp_path / "other.json")
        other.record_last_trading_day(
            "000001.SZ", datetime(2024, 1, 1), reason="已退市"
        )
        registry.merge(other)
        assert registry.is_delisted("000001.SZ", datetime(2024, 1, 1)) is True

    def test_persistence(self, tmp_path):
        path = tmp_path / "delisting_registry.json"
        reg1 = DelistingRegistry(path)
        reg1.record_last_trading_day("000001.SZ", datetime(2024, 1, 1), reason="已退市")
        reg2 = DelistingRegistry(path)
        assert reg2.is_delisted("000001.SZ", datetime(2024, 1, 1)) is True

    def test_resurrection(self, registry):
        registry.record_last_trading_day(
            "000001.SZ", datetime(2024, 1, 1), reason="已退市"
        )
        assert registry.is_delisted("000001.SZ", datetime(2024, 1, 1)) is True
        registry.record_first_trading_day("000001.SZ", datetime(2025, 1, 1))
        assert registry.is_delisted("000001.SZ", datetime(2025, 1, 1)) is False
        lifecycle = registry.get_lifecycle("000001.SZ")
        assert lifecycle.is_active is True
