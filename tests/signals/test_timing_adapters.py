"""app/signals/base.py 择时适配器的契约：输出形状、状态机语义、无未来函数。"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from app.engine.backtest.models import SignalDirection, SignalType
from app.signals.base import (
    StockTimingBollinger,
    StockTimingKDJ,
    StockTimingMA,
    StockTimingMACD,
    StockTimingTurtle,
    StockTimingVolume,
    TimingSignalAdapter,
)

TIMESTAMP = datetime(2026, 1, 5, tzinfo=timezone.utc)
ALL_ADAPTERS: list[TimingSignalAdapter] = [
    StockTimingMA(),
    StockTimingMACD(),
    StockTimingKDJ("金叉"),
    StockTimingBollinger(),
    StockTimingVolume("金叉"),
    StockTimingTurtle(),
]


def _price_frame(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    volume = volumes if volumes is not None else [1_000_000.0] * len(closes)
    return pd.DataFrame({"收盘价_复权": closes, "成交量": volume})


def _bars(*symbols: str) -> dict[str, object]:
    return {symbol: object() for symbol in symbols}


# ── 适配器 → Plum Signal ───────────────────────────────────────────


def test_long_timing_emits_entry_for_each_selected_symbol() -> None:
    adapter = StockTimingMA()
    signals = adapter.to_plum_signals(pd.Series([0, 1]), TIMESTAMP, _bars("A", "B", "C"), ["A", "B"])

    assert [signal.symbol for signal in signals] == ["A", "B"]
    assert all(signal.direction is SignalDirection.LONG for signal in signals)
    assert all(signal.signal_type is SignalType.ENTRY for signal in signals)
    assert all(signal.strength == 1.0 for signal in signals)
    assert all(signal.source == "stock_timing_ma" for signal in signals)
    assert all(signal.timestamp is TIMESTAMP for signal in signals)


def test_flat_timing_emits_exit_for_every_bar_symbol() -> None:
    signals = StockTimingMA().to_plum_signals(pd.Series([1, 0]), TIMESTAMP, _bars("A", "B"))

    assert [signal.symbol for signal in signals] == ["A", "B"]
    assert all(signal.direction is SignalDirection.FLAT for signal in signals)
    assert all(signal.signal_type is SignalType.EXIT for signal in signals)
    assert all(signal.strength == 0.0 for signal in signals)


def test_selected_symbol_absent_from_bars_is_skipped() -> None:
    signals = StockTimingMA().to_plum_signals(pd.Series([1]), TIMESTAMP, _bars("A"), ["A", "Z"])

    assert [signal.symbol for signal in signals] == ["A"]


@pytest.mark.parametrize(
    "timing_result",
    [
        pd.DataFrame({"择时信号": [0, 1], "其他": [9, 9]}),
        pd.DataFrame({"任意列名": [0, 1]}),
    ],
)
def test_dataframe_timing_reads_last_available_row(timing_result: pd.DataFrame) -> None:
    signals = StockTimingMA().to_plum_signals(timing_result, TIMESTAMP, _bars("A"))

    assert [signal.direction for signal in signals] == [SignalDirection.LONG]


# ── 单信号语义 ─────────────────────────────────────────────────────


def test_ma_bias_turns_long_above_threshold_and_flat_below() -> None:
    rising = _price_frame([100.0 + index for index in range(20)])
    falling = _price_frame([100.0 - index for index in range(20)])

    assert StockTimingMA(ma_period=5).compute_timing(rising).iloc[-1] == 1
    assert StockTimingMA(ma_period=5).compute_timing(falling).iloc[-1] == 0


def test_macd_is_long_while_dif_holds_above_signal_line() -> None:
    closes = [100.0 * 1.01**index for index in range(60)]

    assert StockTimingMACD().compute_timing(_price_frame(closes)).iloc[-1] == 1


def test_bollinger_keeps_last_state_between_band_touches() -> None:
    closes = [10.0] * 10 + [20.0] + [20.0] * 5
    signal = StockTimingBollinger(n=4, std_mult=0.5).compute_timing(_price_frame(closes))

    assert signal.iloc[10] == 1
    assert set(signal.iloc[11:].unique()) == {1}


def test_turtle(  ) -> None:
    closes = [10.0, 10.0, 10.0, 10.0, 12.0]
    signal = StockTimingTurtle(n=4).compute_timing(_price_frame(closes))

    assert signal.iloc[:4].tolist() == [0, 0, 0, 0]
    assert signal.iloc[4] == 1


def test_volume_shrink_mode_flags_only_the_shift_before_long_ma_catches_up() -> None:
    volumes = [5_000_000.0] * 5 + [100_000.0] * 10
    signal = StockTimingVolume("缩量", short_n=2, long_n=6).compute_timing(_price_frame([10.0] * 15, volumes))

    assert signal.iloc[:5].sum() == 0
    assert signal.iloc[5] == 1
    assert signal.iloc[-1] == 0


def test_kdj_golden_cross_fires_on_a_single_bar() -> None:
    closes = [10.0 - 0.3 * index for index in range(15)] + [10.0 + index for index in range(15)]
    signal = StockTimingKDJ("金叉").compute_timing(_price_frame(closes))

    assert signal.iloc[15] == 1
    assert signal.drop(index=15).sum() == 0


# ── 边界与无未来函数 ───────────────────────────────────────────────


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda item: item.name)
def test_frame_without_required_columns_returns_empty_series(
    adapter: TimingSignalAdapter,
) -> None:
    assert adapter.compute_timing(pd.DataFrame()).empty


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda item: item.name)
def test_output_has_one_row_per_input_and_no_nan(adapter: TimingSignalAdapter) -> None:
    frame = _price_frame([10.0 + 0.5 * index for index in range(40)])

    signal = adapter.compute_timing(frame)

    assert len(signal) == len(frame)
    assert not signal.isna().any()


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda item: item.name)
def test_prefix_signal_is_unchanged_by_later_data(adapter: TimingSignalAdapter) -> None:
    closes = [10.0 + 0.5 * index for index in range(30)] + [25.0 - 0.8 * index for index in range(30)]
    full = adapter.compute_timing(_price_frame(closes))
    prefix = adapter.compute_timing(_price_frame(closes[:30]))

    pd.testing.assert_series_equal(prefix, full.iloc[:30], check_names=False)
