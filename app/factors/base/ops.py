"""因子运算 — 向量化计算原语。"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class NeutralizeMethod(Enum):
    """中性化方法。"""

    WINSORIZE = "winsorize"
    ZSCORE = "zscore"
    RANK = "rank"


class Ops:
    """使用 NumPy 数组的因子计算运算符。"""

    @staticmethod
    def zscore(values: np.ndarray, skipna: bool = True) -> np.ndarray:
        """Z-Score 标准化。"""
        mean = np.nanmean(values) if skipna else np.mean(values)
        std = np.nanstd(values) if skipna else np.std(values)
        if std == 0:
            return values - mean
        return (values - mean) / std

    @staticmethod
    def rank(values: np.ndarray, ascending: bool = True) -> np.ndarray:
        """排名标准化（百分比排名）。"""
        valid = ~np.isnan(values)
        result = np.full_like(values, np.nan)
        if valid.sum() > 0:
            result[valid] = (
                pd.Series(values[valid]).rank(pct=True, ascending=ascending).values
            )
        return result

    @staticmethod
    def mad(values: np.ndarray, n: float = 1.4826) -> np.ndarray:
        """中位数绝对偏差标准化。"""
        median = np.nanmedian(values)
        mad_val = n * np.nanmedian(np.abs(values - median))
        if mad_val == 0:
            return values - median
        return (values - median) / mad_val

    @staticmethod
    def winsorize_std(values: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """基于标准差的缩尾处理。"""
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0:
            return values
        lower = mean - n_std * std
        upper = mean + n_std * std
        return np.clip(values, lower, upper)

    @staticmethod
    def winsorize_percentile(
        values: np.ndarray, lower: float = 0.01, upper: float = 0.99
    ) -> np.ndarray:
        """基于百分位数的缩尾处理。"""
        lower_val = np.nanpercentile(values, lower * 100)
        upper_val = np.nanpercentile(values, upper * 100)
        return np.clip(values, lower_val, upper_val)

    @staticmethod
    def winsorize_mad(values: np.ndarray, n_mad: float = 3.0) -> np.ndarray:
        """基于 MAD 的缩尾处理。"""
        median = np.nanmedian(values)
        mad_val = np.nanmedian(np.abs(values - median))
        return np.clip(values, median - n_mad * mad_val, median + n_mad * mad_val)

    @staticmethod
    def neutralize(
        factor: np.ndarray,
        groups: np.ndarray,
        method: NeutralizeMethod = NeutralizeMethod.ZSCORE,
    ) -> np.ndarray:
        """按组中性化。"""
        result = factor.copy()
        for g in np.unique(groups[~np.isnan(groups)]):
            mask = groups == g
            result[mask] = factor[mask] - np.nanmean(factor[mask])
        valid = ~np.isnan(result)
        if valid.sum() > 0:
            if method == NeutralizeMethod.ZSCORE:
                std = np.nanstd(result[valid])
                mean = np.nanmean(result[valid])
                result[valid] = (result[valid] - mean) / std if std > 0 else 0
            elif method == NeutralizeMethod.RANK:
                result[valid] = Ops.rank(result[valid])
        return result

    @staticmethod
    def neutralize_market_cap(
        factor: np.ndarray, market_cap: np.ndarray, n_groups: int = 5
    ) -> np.ndarray:
        """按市值分组中性化。"""
        valid = ~(np.isnan(factor) | np.isnan(market_cap))
        if valid.sum() < n_groups:
            return factor
        groups = pd.qcut(market_cap[valid], n_groups, labels=False, duplicates="drop")
        result = factor.copy()
        for g in range(n_groups):
            group_mask = groups == g
            mask = valid & np.asarray(group_mask)
            if mask.sum() > 0:
                result[mask] = factor[mask] - np.nanmean(factor[mask])
        return result

    @staticmethod
    def neutralize_industry(factor: np.ndarray, industry: np.ndarray) -> np.ndarray:
        """按行业中性化。"""
        result = factor.copy()
        for ind in np.unique(industry[~np.isnan(industry)]):
            mask = industry == ind
            if mask.sum() > 0:
                result[mask] = factor[mask] - np.nanmean(factor[mask])
        return result

    @staticmethod
    def rolling_mean(
        values: np.ndarray, window: int, min_periods: int | None = None
    ) -> np.ndarray:
        """滚动均值。"""
        min_periods = min_periods or window // 2
        result = np.full_like(values, np.nan)
        for i in range(window - 1, len(values)):
            window_data = values[i - window + 1 : i + 1]
            if (~np.isnan(window_data)).sum() >= min_periods:
                result[i] = np.nanmean(window_data)
        return result

    @staticmethod
    def rolling_std(
        values: np.ndarray, window: int, min_periods: int | None = None
    ) -> np.ndarray:
        """滚动标准差。"""
        min_periods = min_periods or window // 2
        result = np.full_like(values, np.nan)
        for i in range(window - 1, len(values)):
            window_data = values[i - window + 1 : i + 1]
            if (~np.isnan(window_data)).sum() >= min_periods:
                result[i] = np.nanstd(window_data)
        return result

    @staticmethod
    def ttm_sum(values: np.ndarray, periods: int = 4) -> np.ndarray:
        """TTM 滚动求和。"""
        result = np.full_like(values, np.nan)
        for i in range(periods - 1, len(values)):
            window = values[i - periods + 1 : i + 1]
            if np.any(~np.isnan(window)):
                result[i] = np.nansum(window)
        return result

    @staticmethod
    def ttm_mean(values: np.ndarray, periods: int = 4) -> np.ndarray:
        """TTM 滚动均值。"""
        result = np.full_like(values, np.nan)
        for i in range(periods - 1, len(values)):
            window = values[i - periods + 1 : i + 1]
            if np.any(~np.isnan(window)):
                result[i] = np.nanmean(window)
        return result

    @staticmethod
    def returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
        """收益率计算。"""
        if len(prices) <= periods:
            return np.full_like(prices, np.nan)
        result = np.full_like(prices, np.nan)
        result[periods:] = (prices[periods:] - prices[:-periods]) / prices[:-periods]
        return result

    @staticmethod
    def log_returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
        """对数收益率计算。"""
        if len(prices) <= periods:
            return np.full_like(prices, np.nan)
        result = np.full_like(prices, np.nan)
        valid = prices[:-periods] > 0
        result[periods:][valid] = np.log(
            prices[periods:][valid] / prices[:-periods][valid]
        )
        return result

    @staticmethod
    def momentum(
        values: np.ndarray, window: int = 20, method: str = "cumret"
    ) -> np.ndarray:
        """动量计算。"""
        if method == "cumret":
            return Ops.returns(values, window)
        if method == "zscore":
            return Ops.zscore(Ops.returns(values, 1))
        return values

    @staticmethod
    def combine(
        factors: list[np.ndarray], weights: list[float] | None = None
    ) -> np.ndarray:
        """多因子加权合成。"""
        if not factors:
            return np.array([])
        if weights is None:
            weights = [1.0 / len(factors)] * len(factors)
        weight_arr = np.array(weights)
        weight_arr = weight_arr / weight_arr.sum()
        max_len = max(len(f) for f in factors)
        result = np.zeros(max_len)
        for f, w in zip(factors, weight_arr, strict=True):
            min_len = min(max_len, len(f))
            result[:min_len] += f[:min_len] * w
        return result

    @staticmethod
    def orthogonalize(target: np.ndarray, regressors: list[np.ndarray]) -> np.ndarray:
        """正交化处理。"""
        result = target.copy()
        for reg in regressors:
            if len(reg) != len(target):
                continue
            valid = ~(np.isnan(target) | np.isnan(reg))
            if valid.sum() < 3:
                continue
            t, r = target[valid], reg[valid]
            slope = np.cov(t, r)[0, 1] / np.var(r)
            result[valid] = t - slope * (r - np.mean(r))
        return result


class CrossSectionOps:
    """截面运算符（在每个日期跨股票操作）。"""

    @staticmethod
    def zscore_by_date(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
        """按日期 Z-Score 标准化。"""
        result = df.copy()
        result[value_col] = df.groupby(level="date")[value_col].transform(
            lambda x: Ops.zscore(x.values)
        )
        return result

    @staticmethod
    def rank_by_date(
        df: pd.DataFrame, value_col: str = "value", ascending: bool = True
    ) -> pd.DataFrame:
        """按日期排名标准化。"""
        result = df.copy()
        result[value_col] = df.groupby(level="date")[value_col].transform(
            lambda x: Ops.rank(x.values, ascending)
        )
        return result

    @staticmethod
    def neutralize_by_date(
        df: pd.DataFrame,
        group_col: str,
        value_col: str = "value",
        method: NeutralizeMethod = NeutralizeMethod.ZSCORE,
    ) -> pd.DataFrame:
        """按日期中性化。"""
        result = df.copy()
        result[value_col] = (
            df.groupby(level="date")
            .apply(
                lambda g: pd.Series(
                    Ops.neutralize(g[value_col].values, g[group_col].values, method),
                    index=g.index,
                )
            )
            .droplevel(0)
        )
        return result

    @staticmethod
    def forward_fill(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
        """前向填充。"""
        result = df.copy()
        if "symbol" in result.columns:
            result = result.set_index(["timestamp", "symbol"])
        if value_col in result.columns:
            result[value_col] = result.groupby(level="symbol")[value_col].ffill()
        return result.reset_index()

    @staticmethod
    def resample_to_daily(
        df: pd.DataFrame,
        value_col: str = "value",
        method: str = "last",
    ) -> pd.DataFrame:
        """重采样到日频。"""
        result = df.copy()
        if "symbol" in result.columns:
            result = result.set_index(["timestamp", "symbol"])
        resampler = result.groupby(level="symbol")[value_col]
        if method == "last":
            result = resampler.resample("D").last()
        elif method == "mean":
            result = resampler.resample("D").mean()
        elif method == "sum":
            result = resampler.resample("D").sum()
        return result.reset_index()
