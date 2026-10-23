"""Tests for factor precomputed cache module."""

from datetime import datetime

import numpy as np
import pytest

from app.data.schema import Bar
from app.factors.precomputed import (
    PrecomputedIndicators,
    PrecomputedIndicatorCache,
    compute_for_bars,
    build_precomputed_cache,
)
from app.factors.indicator import IndicatorCalculator
from app.factors.store import FactorMatrix, FactorStore
from app.factors.processor import FactorProcessor


def _make_bars(symbol: str, days: int, base_price: float = 10.0) -> list[Bar]:
    bars = []
    for i in range(days):
        ts = datetime(2024, 1, 1)
        price_int = int((base_price + i * 0.1) * 10_000)
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts,
                open=price_int,
                high=price_int + 100,
                low=price_int - 100,
                close=price_int,
                volume=1000,
                amount=price_int * 1000,
                pre_close=price_int,
            )
        )
    return bars


class TestPrecomputedIndicators:
    def test_at_returns_dict(self):
        n = 100
        data = PrecomputedIndicators(
            closes=np.zeros(n),
            highs=np.zeros(n),
            lows=np.zeros(n),
            ema_fast=np.zeros(n),
            ema_slow=np.zeros(n),
            ma=np.zeros(n),
            atr=np.zeros(n),
            adx=np.zeros(n),
            bb_upper=np.zeros(n),
            bb_middle=np.zeros(n),
            bb_lower=np.zeros(n),
            rsi=np.full(n, 50.0),
        )
        result = data.at(50)
        assert "price" in result
        assert "rsi" in result
        assert result["rsi"] == 50.0

    def test_n_property(self):
        n = 50
        data = PrecomputedIndicators(
            closes=np.zeros(n),
            highs=np.zeros(n),
            lows=np.zeros(n),
            ema_fast=np.zeros(n),
            ema_slow=np.zeros(n),
            ma=np.zeros(n),
            atr=np.zeros(n),
            adx=np.zeros(n),
            bb_upper=np.zeros(n),
            bb_middle=np.zeros(n),
            bb_lower=np.zeros(n),
            rsi=np.zeros(n),
        )
        assert data.n == n


class TestPrecomputedIndicatorCache:
    def test_get_set(self):
        cache = PrecomputedIndicatorCache()
        indicators = PrecomputedIndicators(
            closes=np.zeros(10),
            highs=np.zeros(10),
            lows=np.zeros(10),
            ema_fast=np.zeros(10),
            ema_slow=np.zeros(10),
            ma=np.zeros(10),
            atr=np.zeros(10),
            adx=np.zeros(10),
            bb_upper=np.zeros(10),
            bb_middle=np.zeros(10),
            bb_lower=np.zeros(10),
            rsi=np.zeros(10),
        )
        cache.set("sh.600519", indicators)
        assert cache.get("sh.600519") is indicators
        assert cache.get("sh.000001") is None

    def test_contains(self):
        cache = PrecomputedIndicatorCache()
        indicators = PrecomputedIndicators(
            closes=np.zeros(10),
            highs=np.zeros(10),
            lows=np.zeros(10),
            ema_fast=np.zeros(10),
            ema_slow=np.zeros(10),
            ma=np.zeros(10),
            atr=np.zeros(10),
            adx=np.zeros(10),
            bb_upper=np.zeros(10),
            bb_middle=np.zeros(10),
            bb_lower=np.zeros(10),
            rsi=np.zeros(10),
        )
        cache.set("sh.600519", indicators)
        assert "sh.600519" in cache
        assert "sh.000001" not in cache


class TestComputeForBars:
    def test_compute_indicators(self):
        bars = _make_bars("sh.600519", 250)
        result = compute_for_bars(
            bars,
            ema_fast_period=14,
            ema_slow_period=25,
            ma_trend_period=200,
            atr_period=14,
            adx_period=14,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14,
        )
        assert result.n == 250
        assert len(result.closes) == 250
        assert len(result.atr) == 250
        assert len(result.rsi) == 250

    def test_compute_at_index(self):
        bars = _make_bars("sh.600519", 250)
        result = compute_for_bars(
            bars,
            ema_fast_period=14,
            ema_slow_period=25,
            ma_trend_period=200,
            atr_period=14,
            adx_period=14,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14,
        )
        values = result.at(200)
        assert "price" in values
        assert "ema_fast" in values
        assert "atr" in values
        assert "rsi" in values

    def test_empty_bars_raises(self):
        with pytest.raises(ValueError, match="Cannot compute indicators"):
            compute_for_bars(
                [],
                ema_fast_period=14,
                ema_slow_period=25,
                ma_trend_period=200,
                atr_period=14,
                adx_period=14,
                bb_period=20,
                bb_std=2.0,
                rsi_period=14,
            )


class TestBuildPrecomputedCache:
    def test_build_for_multiple_symbols(self):
        all_data = {
            "sh.600519": _make_bars("sh.600519", 250),
            "sh.600036": _make_bars("sh.600036", 250, base_price=20.0),
        }
        cache = build_precomputed_cache(
            all_data,
            ema_fast_period=14,
            ema_slow_period=25,
            ma_trend_period=200,
            atr_period=14,
            adx_period=14,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14,
        )
        assert "sh.600519" in cache
        assert "sh.600036" in cache
        assert cache.get("sh.600519").n == 250

    def test_skips_empty_symbols(self):
        all_data = {
            "sh.600519": _make_bars("sh.600519", 250),
            "sh.000001": [],
        }
        cache = build_precomputed_cache(
            all_data,
            ema_fast_period=14,
            ema_slow_period=25,
            ma_trend_period=200,
            atr_period=14,
            adx_period=14,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14,
        )
        assert "sh.600519" in cache
        assert "sh.000001" not in cache


class TestIndicatorCalculator:
    def test_update_and_calculate(self):
        calc = IndicatorCalculator(
            ema_fast_period=14,
            ema_slow_period=25,
            ma_trend_period=200,
            atr_period=14,
            adx_period=14,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14,
        )
        bars = _make_bars("sh.600519", 250)
        for bar in bars:
            calc.update("sh.600519", bar)

        result = calc.calculate("sh.600519", warmup_bars=200)
        assert result is not None
        assert "price" in result
        assert "ema_fast" in result
        assert "atr" in result

    def test_precomputed_path(self):
        calc = IndicatorCalculator(
            ema_fast_period=14,
            ema_slow_period=25,
            ma_trend_period=20,
            atr_period=14,
            adx_period=14,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14,
        )
        bars = _make_bars("sh.600519", 250)
        for bar in bars:
            calc.update("sh.600519", bar)

        cache = build_precomputed_cache(
            {"sh.600519": bars},
            ema_fast_period=14,
            ema_slow_period=25,
            ma_trend_period=20,
            atr_period=14,
            adx_period=14,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14,
        )
        calc.attach_precomputed(cache)

        result = calc.calculate("sh.600519", warmup_bars=200)
        assert result is not None
        assert "price" in result
        assert "bb_upper" in result

    def test_unknown_symbol_returns_none(self):
        calc = IndicatorCalculator()
        result = calc.calculate("sh.000001", warmup_bars=20)
        assert result is None

    def test_insufficient_warmup_returns_none(self):
        calc = IndicatorCalculator()
        bars = _make_bars("sh.600519", 10)
        for bar in bars:
            calc.update("sh.600519", bar)
        result = calc.calculate("sh.600519", warmup_bars=100)
        assert result is None

    def test_get_set_state(self):
        calc = IndicatorCalculator()
        bars = _make_bars("sh.600519", 50)
        for bar in bars:
            calc.update("sh.600519", bar)

        state = calc.get_state("sh.600519")
        assert state is not None
        assert "history" in state
        assert "ema_fast" in state

        calc2 = IndicatorCalculator()
        calc2.set_state("sh.600519", state)
        restored = calc2.get_state("sh.600519")
        assert restored is not None
        assert len(restored["history"]) == 50


class TestFactorMatrix:
    def test_score_at(self):
        idx = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        df = __import__("pandas").DataFrame(
            {"sh.600519": [1.0, 2.0], "sh.600036": [3.0, 4.0]},
            index=idx,
        )
        matrix = FactorMatrix(name="TEST", data=df)
        scores = matrix.score_at(datetime(2024, 1, 1))
        assert scores["sh.600519"] == 1.0
        assert scores["sh.600036"] == 3.0

    def test_score_at_missing_date(self):
        idx = [datetime(2024, 1, 1)]
        df = __import__("pandas").DataFrame(
            {"sh.600519": [1.0]},
            index=idx,
        )
        matrix = FactorMatrix(name="TEST", data=df)
        scores = matrix.score_at(datetime(2024, 1, 2))
        assert scores == {}

    def test_ranked_at(self):
        idx = [datetime(2024, 1, 1)]
        df = __import__("pandas").DataFrame(
            {"sh.600519": [3.0], "sh.600036": [1.0], "sh.600000": [2.0]},
            index=idx,
        )
        matrix = FactorMatrix(name="TEST", data=df)
        ranked = matrix.ranked_at(datetime(2024, 1, 1))
        assert ranked[0][0] == "sh.600519"
        assert ranked[-1][0] == "sh.600036"

    def test_ranked_at_top_n(self):
        idx = [datetime(2024, 1, 1)]
        df = __import__("pandas").DataFrame(
            {"A": [1.0], "B": [2.0], "C": [3.0]},
            index=idx,
        )
        matrix = FactorMatrix(name="TEST", data=df)
        ranked = matrix.ranked_at(datetime(2024, 1, 1), top_n=2)
        assert len(ranked) == 2

    def test_ranked_at_ascending(self):
        idx = [datetime(2024, 1, 1)]
        df = __import__("pandas").DataFrame(
            {"A": [3.0], "B": [1.0], "C": [2.0]},
            index=idx,
        )
        matrix = FactorMatrix(name="TEST", data=df)
        ranked = matrix.ranked_at(datetime(2024, 1, 1), ascending=True)
        assert ranked[0][0] == "B"


class TestFactorStore:
    def test_register_and_get(self):
        store = FactorStore()
        idx = [datetime(2024, 1, 1)]
        df = __import__("pandas").DataFrame(
            {"sh.600519": [1.0]},
            index=idx,
        )
        matrix = FactorMatrix(name="TEST", data=df)
        store.register(matrix)
        assert store.get("TEST") is matrix

    def test_score_at_single_factor(self):
        store = FactorStore()
        idx = [datetime(2024, 1, 1)]
        df = __import__("pandas").DataFrame(
            {"sh.600519": [1.0], "sh.600036": [2.0]},
            index=idx,
        )
        store.register(FactorMatrix(name="F1", data=df))
        scores = store.score_at(datetime(2024, 1, 1), factor_names=["F1"])
        assert scores["sh.600519"] == 1.0

    def test_score_at_multi_factor(self):
        store = FactorStore()
        idx = [datetime(2024, 1, 1)]
        df1 = __import__("pandas").DataFrame(
            {"sh.600519": [1.0]},
            index=idx,
        )
        df2 = __import__("pandas").DataFrame(
            {"sh.600519": [3.0]},
            index=idx,
        )
        store.register(FactorMatrix(name="F1", data=df1))
        store.register(FactorMatrix(name="F2", data=df2))
        scores = store.score_at(datetime(2024, 1, 1), factor_names=["F1", "F2"])
        assert scores["sh.600519"] == 2.0


class TestFactorProcessor:
    def test_winsorize_std(self):
        series = __import__("pandas").Series([1.0, 2.0, 3.0, 100.0])
        result = FactorProcessor.winsorize_std(series, n_std=1.0)
        assert result.iloc[-1] < 100.0

    def test_winsorize_percentile(self):
        series = __import__("pandas").Series(list(range(100)) + [10000.0])
        result = FactorProcessor.winsorize_percentile(series, lower=0.01, upper=0.99)
        assert result.iloc[-1] < 10000.0

    def test_standardize(self):
        series = __import__("pandas").Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = FactorProcessor.standardize(series)
        assert abs(result.mean()) < 1e-10

    def test_combine_equal_weights(self):
        f1 = __import__("pandas").Series([1.0, 2.0, 3.0])
        f2 = __import__("pandas").Series([3.0, 4.0, 5.0])
        result = FactorProcessor.combine([f1, f2])
        expected = __import__("pandas").Series([2.0, 3.0, 4.0])
        __import__("numpy").testing.assert_array_almost_equal(
            result.values, expected.values
        )

    def test_combine_weighted(self):
        f1 = __import__("pandas").Series([1.0, 2.0, 3.0])
        f2 = __import__("pandas").Series([3.0, 4.0, 5.0])
        result = FactorProcessor.combine([f1, f2], weights=[0.3, 0.7])
        expected = __import__("pandas").Series([2.4, 3.4, 4.4])
        __import__("numpy").testing.assert_array_almost_equal(
            result.values, expected.values
        )

    def test_neutralize_by_market_cap(self):
        factor = __import__("pandas").Series([1.0, 2.0, 3.0, 4.0, 5.0])
        mcap = __import__("pandas").Series([100.0, 100.0, 200.0, 200.0, 200.0])
        result = FactorProcessor.neutralize_by_market_cap(factor, mcap, n_groups=2)
        assert abs(result.mean()) < 1e-10

    def test_neutralize_by_industry(self):
        factor = __import__("pandas").Series([1.0, 2.0, 3.0, 4.0])
        industry = __import__("pandas").Series(["A", "A", "B", "B"])
        result = FactorProcessor.neutralize_by_industry(factor, industry)
        assert abs(result.mean()) < 1e-10
