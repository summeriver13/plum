"""因子预处理器 — 去极值、标准化、中性化。"""

from __future__ import annotations

import pandas as pd


class FactorProcessor:
    """多因子策略的因子预处理算子。"""

    @staticmethod
    def winsorize_std(
        series: pd.Series,
        n_std: float = 3.0,
    ) -> pd.Series:
        """使用 N 倍标准差进行缩尾处理。

        参数:
            series: 输入因子序列
            n_std: 标准差倍数（默认 3.0）

        返回:
            缩尾后的序列
        """
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series
        lower = mean - n_std * std
        upper = mean + n_std * std
        return series.clip(lower, upper)

    @staticmethod
    def winsorize_percentile(
        series: pd.Series,
        lower: float = 0.01,
        upper: float = 0.99,
    ) -> pd.Series:
        """使用百分位数进行缩尾处理。

        参数:
            series: 输入因子序列
            lower: 下百分位（默认 1%）
            upper: 上百分位（默认 99%）

        返回:
            缩尾后的序列
        """
        return series.clip(series.quantile(lower), series.quantile(upper))

    @staticmethod
    def standardize(series: pd.Series) -> pd.Series:
        """Z-score 标准化。

        参数:
            series: 输入因子序列

        返回:
            标准化序列（均值=0，标准差=1）
        """
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series - mean
        return (series - mean) / std

    @staticmethod
    def neutralize_by_market_cap(
        factor: pd.Series,
        market_cap: pd.Series,
        n_groups: int = 5,
    ) -> pd.Series:
        """按市值分组进行中性化处理。

        通过减去组均值去除风格偏差。
        分组按市值五等分定义。

        参数:
            factor: 需要中性化的因子
            market_cap: 市值序列（用于分组）
            n_groups: 分组数量（默认 5 为五等分）

        返回:
            中性化后的因子
        """
        aligned_factor = factor.copy()
        aligned_mcap = market_cap.reindex(factor.index)

        groups = pd.qcut(aligned_mcap, n_groups, labels=False, duplicates="drop")
        group_means = aligned_factor.groupby(groups).transform("mean")

        return aligned_factor - group_means

    @staticmethod
    def neutralize_by_industry(
        factor: pd.Series,
        industry: pd.Series,
    ) -> pd.Series:
        """按行业进行中性化处理。

        参数:
            factor: 需要中性化的因子
            industry: 行业代码序列

        返回:
            行业中性化后的因子
        """
        aligned_factor = factor.copy()
        aligned_industry = industry.reindex(factor.index)

        industry_means = aligned_factor.groupby(aligned_industry).transform("mean")

        return aligned_factor - industry_means

    @staticmethod
    def combine(
        factors: list[pd.Series],
        weights: list[float] | None = None,
    ) -> pd.Series:
        """按权重组合多个因子。

        参数:
            factors: 因子序列列表
            weights: 权重列表（默认等权重）

        返回:
            组合后的因子序列
        """
        if not factors:
            return pd.Series(dtype=float)

        aligned_factors = [f.reindex(factors[0].index) for f in factors]
        combined = pd.concat(aligned_factors, axis=1)

        if weights is None:
            weights = [1.0 / len(factors)] * len(factors)

        result = pd.Series(0.0, index=combined.index)
        for w, col in zip(weights, combined.columns, strict=True):
            result = result + w * combined[col]
        return result
