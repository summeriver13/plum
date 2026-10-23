"""退市注册表构建脚本（scripts/build_delisting_registry.py）测试。

覆盖：
- 数据湖代码 → plum 代码转换
- 退市/存活/退市日缺失三种记录的字段生成
- 生成文件可被 DelistingRegistry 正常加载（is_delisted 时点语义）
- 引擎股票池预过滤：只剔除「回测开始前已退市」，保留「尚未上市」
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pandas as pd

from app.data.delisting_registry import DelistingRegistry
from app.engine.backtest.engine import Engine
from app.engine.backtest.portfolio import Portfolio
from app.engine.backtest.scenarios.commission import AShareCommissionModel
from scripts.build_delisting_registry import (
    build_record,
    to_plum_symbol,
    write_registry,
)


def make_row(**overrides: object) -> pd.Series:
    """构造 instruments 行（默认：已退市的拉夏贝尔）。"""
    base: dict[str, object] = {
        "symbol": "603157.SH",
        "is_delisted": True,
        "ipo_date": pd.Timestamp("2017-09-25"),
        "out_date": pd.Timestamp("2022-05-24"),
    }
    base.update(overrides)
    return pd.Series(base)


class TestSymbolConversion:
    def test_to_plum_symbol(self) -> None:
        assert to_plum_symbol("603157.SH") == "SH.603157"
        assert to_plum_symbol("000001.SZ") == "SZ.000001"


class TestBuildRecord:
    def test_delisted_record(self) -> None:
        symbol, record = build_record(make_row(), pd.Timestamp("2022-05-16"))
        assert symbol == "SH.603157"
        assert record["first_trading_day"] == "2017-09-25"
        assert record["last_trading_day"] == "2022-05-24"
        assert record["delisting_date"] == "2022-05-24"
        assert record["is_active"] is False

    def test_out_date_missing_falls_back_to_last_bar(self) -> None:
        _, record = build_record(make_row(out_date=pd.NaT), pd.Timestamp("2022-05-16"))
        assert record["last_trading_day"] == "2022-05-16"

    def test_active_record_has_no_last_day(self) -> None:
        _, record = build_record(make_row(is_delisted=False, out_date=pd.NaT), None)
        assert record["is_active"] is True
        assert "last_trading_day" not in record


class TestRegistryRoundTrip:
    def test_generated_file_loads_with_pit_semantics(self, tmp_path) -> None:
        records = {
            "SH.603157": {
                "symbol": "SH.603157",
                "first_trading_day": "2017-09-25",
                "last_trading_day": "2022-05-24",
                "delisting_date": "2022-05-24",
                "reason": "退市",
                "is_active": False,
            }
        }
        path = tmp_path / "delisting_registry.json"
        write_registry(records, path)
        registry = DelistingRegistry(path)
        assert registry.is_delisted("SH.603157", datetime(2022, 5, 24)) is True
        assert registry.is_delisted("SH.603157", datetime(2022, 5, 23)) is False


class TestEngineUniversePreFilter:
    """预过滤必须保留「回测后才上市」的标的。

    历史缺陷：用 UniverseFilter.is_blocked(s, "BUY", start) 会把尚未上市的
    标的判为不可买而整只剔除，使其上市后永远无法进入回测。
    """

    def _engine(self, tmp_path) -> Engine:
        return Engine(
            portfolio=Portfolio(initial_cash=Decimal("1000000")),
            catalog=None,
            commission=AShareCommissionModel(),
            delisting_registry=DelistingRegistry(tmp_path / "reg.json"),
        )

    def test_keeps_not_yet_listed_and_drops_delisted_before_start(
        self, tmp_path
    ) -> None:
        engine = self._engine(tmp_path)
        registry = engine.delisting_registry
        registry.record_last_trading_day("SH.600001", datetime(2019, 1, 2))
        registry.record_first_trading_day("SZ.301589", datetime(2024, 2, 8))

        active = engine._filter_active_symbols(
            ["SH.600001", "SZ.301589", "SH.600519"], datetime(2020, 1, 2)
        )

        assert active == ["SZ.301589", "SH.600519"]

    def test_disabled_protection_returns_all(self, tmp_path) -> None:
        engine = self._engine(tmp_path)
        engine.enable_delisting_protection = False
        symbols = ["SH.600001", "SH.600519"]
        assert engine._filter_active_symbols(symbols, datetime(2020, 1, 2)) == symbols
