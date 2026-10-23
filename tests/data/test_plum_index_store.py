"""指数（基准）落盘往返测试：构建脚本写出 → MarketCatalog 能按符号读回。

覆盖的核心契约：
- 目录桶为 `IDX`，plum 符号 `IDX.<6位代码>`；
- timestamp 与个股同约定（交易日 - 1 天 + 16:00），否则基准按日期对齐会静默回退；
- `IDX` 不进入股票池扫描（SH/SZ 白名单之外）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_plum_market_from_qdata import build_index_frame, write_indexes  # noqa: E402

INDEX_SYMBOL = "000852.SH"
TRADING_DAY = pd.Timestamp("2024-01-02")


def _index_rows() -> pd.DataFrame:
    """构造两日中证1000 指数行（契约列）。"""
    return pd.DataFrame(
        {
            "symbol": [INDEX_SYMBOL, INDEX_SYMBOL],
            "datetime": [TRADING_DAY, pd.Timestamp("2024-01-03")],
            "open": [7400.0, 7420.0],
            "high": [7460.0, 7450.0],
            "low": [7380.0, 7390.0],
            "close": [7427.2405, 7410.0],
            "pre_close": [7587.0, 7427.2405],
            "volume": [24958630000, 21000000000],
            "amount": [3.6e11, 3.1e11],
            "pct_change": [-2.115404, -0.23],
        }
    )


def test_build_index_frame_uses_stock_timestamp_convention() -> None:
    """timestamp = 交易日 - 1 天 + 16:00，与个股完全一致。"""
    frame = build_index_frame(_index_rows())
    expected = TRADING_DAY - pd.Timedelta(days=1) + pd.Timedelta(hours=16)
    assert frame.index[0] == expected
    assert frame.index.name == "timestamp"


def test_build_index_frame_fills_missing_concepts() -> None:
    """指数无复权/停牌概念：对应列填默认值，缺失估值列留空。"""
    frame = build_index_frame(_index_rows())
    assert frame["adj_factor"].tolist() == [1.0, 1.0]
    assert frame["is_st"].tolist() == [False, False]
    assert frame["is_delisted"].tolist() == [False, False]
    assert frame["close"].tolist() == [7427.2405, 7410.0]


def test_build_index_frame_drops_rows_without_prices() -> None:
    """缺价行丢弃（避免 Decimal('nan') 污染 Bar 构造）。"""
    rows = _index_rows()
    rows.loc[1, "close"] = None
    assert len(build_index_frame(rows)) == 1


def test_write_indexes_round_trips_through_catalog(tmp_path: Path) -> None:
    """写出 → MarketCatalog 按 IDX.000852 读回，价格与日期一致。"""
    from app.data.store.market import MarketCatalog

    # catalog 根为 market 目录，周期目录 1d 在其下（与线上 data/market/1d 一致）
    write_indexes(_index_rows(), tmp_path / "1d")
    written = tmp_path / "1d" / "IDX" / "000852.parquet"
    assert written.exists()

    catalog = MarketCatalog(catalog_root=tmp_path)
    bars = catalog.load_bars("IDX.000852", "1d")
    assert len(bars) == 2
    assert bars[-1].close == pytest.approx(7410.0)
    assert bars[0].timestamp == (
        TRADING_DAY - pd.Timedelta(days=1) + pd.Timedelta(hours=16)
    )


def test_index_bucket_is_excluded_from_symbol_listing(tmp_path: Path) -> None:
    """IDX 不在股票池扫描结果内（符号白名单只有 SH/SZ/BJ）。"""
    from app.data.store.market import MarketCatalog

    write_indexes(_index_rows(), tmp_path / "1d")
    stocks = tmp_path / "1d" / "SH"
    stocks.mkdir()
    _index_rows().drop(columns=["symbol"]).to_parquet(stocks / "600519.parquet")

    symbols = MarketCatalog(catalog_root=tmp_path).list_symbols("1d")
    assert "IDX.000852" not in symbols
    assert symbols == ["SH.600519"]


class TestStoreBuildConsistency:
    """构建脚本的副本一致性抽检（防静默丢值）。"""

    def test_verify_sample_detects_zeroed_amount(self, tmp_path: Path) -> None:
        """湖里有成交额、副本被写成 0 → 必须报异常。"""
        import pandas as pd

        from scripts.build_plum_market_from_qdata import verify_sample

        bars = pd.DataFrame(
            {
                "symbol": ["600000.SH"] * 3,
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
                "amount": [100.0, 200.0, 300.0],
            }
        )
        store = tmp_path / "SH"
        store.mkdir()
        pd.DataFrame({"amount": [100.0, 0.0, 0.0]}).to_parquet(store / "600000.parquet")

        assert verify_sample(tmp_path, bars, {}, sample_every=1) == 1

    def test_verify_sample_passes_on_faithful_copy(self, tmp_path: Path) -> None:
        """忠实副本（成交额一致、circ_mv 覆盖一致）→ 无异常。"""
        import pandas as pd

        from scripts.build_plum_market_from_qdata import verify_sample

        bars = pd.DataFrame(
            {
                "symbol": ["600000.SH"] * 2,
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                "amount": [100.0, 200.0],
            }
        )
        store = tmp_path / "SH"
        store.mkdir()
        pd.DataFrame({"amount": [100.0, 200.0], "circ_mv": [None, 5.0]}).to_parquet(
            store / "600000.parquet"
        )
        valuation = pd.DataFrame(
            {"circ_mv": [None, 5.0]},
            index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]),
        )

        assert (
            verify_sample(tmp_path, bars, {"600000.SH": valuation}, sample_every=1) == 0
        )

    def test_verify_sample_detects_missing_circ_mv(self, tmp_path: Path) -> None:
        """湖里有流通市值、副本全空 → 必须报异常（历史缺陷形态）。"""
        import pandas as pd

        from scripts.build_plum_market_from_qdata import verify_sample

        bars = pd.DataFrame(
            {
                "symbol": ["SH.600000"],
                "datetime": pd.to_datetime(["2020-01-02"]),
                "amount": [100.0],
            }
        )
        store = tmp_path / "SH"
        store.mkdir()
        pd.DataFrame({"amount": [100.0], "circ_mv": [None]}).to_parquet(
            store / "600000.parquet"
        )
        valuation = pd.DataFrame(
            {"circ_mv": [5.0]}, index=pd.DatetimeIndex(["2020-01-02"])
        )

        assert (
            verify_sample(tmp_path, bars, {"600000.SH": valuation}, sample_every=1) == 1
        )
