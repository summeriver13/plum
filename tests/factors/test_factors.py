"""Tests for factor module."""

from datetime import datetime

import numpy as np
import pandas as pd

from app.factors.base import (
    Factor,
    FactorCategory,
    DataBundle,
    FactorRegistry,
    FactorDAG,
    Ops,
    CrossSectionOps,
    register_factor,
    clear_registry,
)
from app.factors.engine import FactorEngine
from app.factors.universe import Universe, UniverseManager, UniverseRecord


class TestOps:
    def test_zscore(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = Ops.zscore(values)
        assert np.isclose(np.nanmean(result), 0.0, atol=1e-10)
        assert np.isclose(np.nanstd(result), 1.0, atol=1e-10)

    def test_zscore_with_nan(self):
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = Ops.zscore(values)
        assert np.isnan(result[1])

    def test_rank(self):
        values = np.array([10.0, 30.0, 20.0])
        result = Ops.rank(values)
        assert result[0] < result[2] < result[1]

    def test_winsorize_std(self):
        values = np.array([1.0, 2.0, 3.0, 200.0])
        result = Ops.winsorize_std(values, n_std=1.0)
        assert result[-1] < 200.0

    def test_returns(self):
        prices = np.array([100.0, 110.0, 120.0])
        result = Ops.returns(prices, periods=1)
        assert np.isnan(result[0])
        assert np.isclose(result[1], 0.10)

    def test_rolling_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = Ops.rolling_mean(values, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isclose(result[2], 2.0)

    def test_combine_equal_weights(self):
        f1 = np.array([1.0, 2.0, 3.0])
        f2 = np.array([4.0, 5.0, 6.0])
        result = Ops.combine([f1, f2])
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_momentum(self):
        prices = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        result = Ops.momentum(prices, window=2, method="cumret")
        assert np.isclose(result[2], 0.10)


class TestCrossSectionOps:
    def test_zscore_by_date(self):
        idx = pd.MultiIndex.from_tuples(
            [
                (datetime(2024, 1, 1), "A"),
                (datetime(2024, 1, 1), "B"),
                (datetime(2024, 1, 1), "C"),
            ],
            names=["date", "symbol"],
        )
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=idx)
        result = CrossSectionOps.zscore_by_date(df)
        assert np.isclose(result["value"].mean(), 0.0, atol=1e-10)


class TestFactorRegistry:
    def setup_method(self):
        clear_registry()

    def teardown_method(self):
        clear_registry()

    def test_register_and_get(self):
        @register_factor
        class TestFactor(Factor):
            name = "TEST_FACTOR"
            category = FactorCategory.VALUATION
            indicator = "TEST"
            period = "1D"
            processing = "RAW"
            description = "Test factor"
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                return pd.DataFrame(columns=["value"])

        registry = FactorRegistry.get_instance()
        assert registry.get("TEST_FACTOR") is TestFactor
        assert registry.count == 1

    def test_list_all(self):
        @register_factor
        class TestFactorA(Factor):
            name = "TEST_A"
            category = FactorCategory.VALUATION
            indicator = "A"
            period = "1D"
            processing = "RAW"
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                return pd.DataFrame(columns=["value"])

        registry = FactorRegistry.get_instance()
        assert "TEST_A" in registry.list_all()

    def test_list_by_category(self):
        @register_factor
        class TestValFactor(Factor):
            name = "TEST_VAL"
            category = FactorCategory.VALUATION
            indicator = "VAL"
            period = "1D"
            processing = "RAW"
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                return pd.DataFrame(columns=["value"])

        registry = FactorRegistry.get_instance()
        assert "TEST_VAL" in registry.list_by_category(FactorCategory.VALUATION)

    def test_get_factor_instance(self):
        @register_factor
        class TestInstFactor(Factor):
            name = "TEST_INST"
            category = FactorCategory.QUALITY
            indicator = "INST"
            period = "1D"
            processing = "RAW"
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                return pd.DataFrame(columns=["value"])

        registry = FactorRegistry.get_instance()
        instance = registry.get_factor_instance("TEST_INST")
        assert instance is not None
        assert instance.name == "TEST_INST"


class TestFactorDAG:
    def setup_method(self):
        clear_registry()

    def teardown_method(self):
        clear_registry()

    def test_topological_sort(self):
        @register_factor
        class IndepFactor(Factor):
            name = "INDEP"
            category = FactorCategory.SIZE
            indicator = "INDEP"
            period = "1D"
            processing = "RAW"
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                return pd.DataFrame(columns=["value"])

        @register_factor
        class DepFactor(Factor):
            name = "DEP"
            category = FactorCategory.COMPOSITE
            indicator = "DEP"
            period = "1D"
            processing = "RAW"
            dependencies = ["INDEP"]
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                return pd.DataFrame(columns=["value"])

        registry = FactorRegistry.get_instance()
        dag = FactorDAG(registry)
        order = dag.get_computation_order(["INDEP", "DEP"])
        assert order.index("INDEP") < order.index("DEP")

    def test_detect_cycle(self):
        @register_factor
        class CycleA(Factor):
            name = "CYCLE_A"
            category = FactorCategory.SIZE
            indicator = "A"
            period = "1D"
            processing = "RAW"
            dependencies = ["CYCLE_B"]
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                return pd.DataFrame(columns=["value"])

        @register_factor
        class CycleB(Factor):
            name = "CYCLE_B"
            category = FactorCategory.SIZE
            indicator = "B"
            period = "1D"
            processing = "RAW"
            dependencies = ["CYCLE_A"]
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                return pd.DataFrame(columns=["value"])

        registry = FactorRegistry.get_instance()
        dag = FactorDAG(registry)
        dag.build(["CYCLE_A", "CYCLE_B"])
        cycle = dag.detect_cycle()
        assert cycle is not None


class TestFactorEngine:
    def setup_method(self):
        clear_registry()

    def teardown_method(self):
        clear_registry()

    def test_compute_single(self):
        @register_factor
        class SimpleFactor(Factor):
            name = "SIMPLE"
            category = FactorCategory.VALUATION
            indicator = "SIMPLE"
            period = "1D"
            processing = "RAW"
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                idx = pd.MultiIndex.from_tuples(
                    [
                        (datetime(2024, 1, 1), "A"),
                        (datetime(2024, 1, 1), "B"),
                    ],
                    names=["timestamp", "symbol"],
                )
                return pd.DataFrame({"value": [1.0, 2.0]}, index=idx)

        engine = FactorEngine()
        result = engine.compute_single("SIMPLE", DataBundle())
        assert "value" in result.columns
        assert len(result) == 2

    def test_cache(self):
        @register_factor
        class CachedFactor(Factor):
            name = "CACHED"
            category = FactorCategory.MOMENTUM
            indicator = "CACHED"
            period = "1D"
            processing = "RAW"
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                idx = pd.MultiIndex.from_tuples(
                    [
                        (datetime(2024, 1, 1), "A"),
                    ],
                    names=["timestamp", "symbol"],
                )
                return pd.DataFrame({"value": [1.0]}, index=idx)

        engine = FactorEngine()
        result1 = engine.compute_single("CACHED", DataBundle())
        result2 = engine.compute_single("CACHED", DataBundle())
        assert result1 is result2

    def test_clear_cache(self):
        @register_factor
        class CacheClearFactor(Factor):
            name = "CACHE_CLEAR"
            category = FactorCategory.VOLATILITY
            indicator = "CC"
            period = "1D"
            processing = "RAW"
            data_source = "Test"

            def compute(self, data_bundle, pit_context=None):
                idx = pd.MultiIndex.from_tuples(
                    [
                        (datetime(2024, 1, 1), "A"),
                    ],
                    names=["timestamp", "symbol"],
                )
                return pd.DataFrame({"value": [1.0]}, index=idx)

        engine = FactorEngine()
        engine.compute_single("CACHE_CLEAR", DataBundle())
        engine.clear_cache()
        assert "CACHE_CLEAR" not in engine._cache


class TestUniverseManager:
    def test_add_and_get_snapshot(self):
        manager = UniverseManager()
        universe = Universe(
            date=datetime(2024, 1, 1),
            symbols={"000001.SZ", "600000.SH"},
            records=[
                UniverseRecord("000001.SZ", "平安银行", "1", datetime(2024, 1, 1)),
                UniverseRecord("600000.SH", "浦发银行", "1", datetime(2024, 1, 1)),
            ],
        )
        manager.add_snapshot(universe)
        result = manager.get_universe(datetime(2024, 1, 1))
        assert result is not None
        assert "000001.SZ" in result.symbols

    def test_get_nearest_snapshot(self):
        manager = UniverseManager()
        manager.add_snapshot(
            Universe(
                date=datetime(2024, 1, 1),
                symbols={"A", "B"},
            )
        )
        manager.add_snapshot(
            Universe(
                date=datetime(2024, 2, 1),
                symbols={"A", "B", "C"},
            )
        )
        result = manager.get_universe(datetime(2024, 1, 15))
        assert result is not None
        assert result.date == datetime(2024, 1, 1)

    def test_filter_tradable(self):
        manager = UniverseManager()
        symbols = ["000001.SZ", "688001.SH", "300001.SZ", "*ST001.SZ"]
        result = manager.filter_tradable(
            symbols, capital=200_000, experience_years=2, exclude_st=True
        )
        assert "000001.SZ" in result
        assert "688001.SH" not in result
        assert "300001.SZ" in result
        assert "*ST001.SZ" not in result
