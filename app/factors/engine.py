"""因子计算引擎。

负责因子的调度执行、依赖解析和结果缓存。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.factors.base.factor import DataBundle
from app.factors.base.registry import FactorDAG, FactorRegistry


class FactorEngine:
    """因子计算引擎。

    功能：
    - 按依赖顺序执行因子计算
    - 管理数据供给
    - 缓存中间结果
    """

    def __init__(self, registry: FactorRegistry | None = None) -> None:
        self._registry = registry or FactorRegistry.get_instance()
        self._cache: dict[str, pd.DataFrame] = {}

    def clear_cache(self) -> None:
        """清除计算缓存。"""
        self._cache.clear()

    def compute(
        self,
        factor_names: list[str],
        data_bundle: DataBundle,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """计算指定因子。

        参数:
            factor_names: 要计算的因子名称列表
            data_bundle: 输入数据
            symbols: 股票列表
            start: 开始日期
            end: 结束日期

        返回:
            {因子名称: DataFrame} 字典
        """
        dag = FactorDAG(self._registry)
        order = dag.get_computation_order(factor_names)

        results: dict[str, pd.DataFrame] = {}

        for name in order:
            if name in self._cache:
                results[name] = self._cache[name]
                continue

            factor = self._registry.get_factor_instance(name)
            if factor is None:
                continue

            df = factor.compute(data_bundle)
            factor.validate(df)
            results[name] = df
            self._cache[name] = df

        return results

    def compute_single(
        self,
        factor_name: str,
        data_bundle: DataBundle,
    ) -> pd.DataFrame:
        """计算单个因子。"""
        factor = self._registry.get_factor_instance(factor_name)
        if factor is None:
            raise ValueError(f"未找到因子: {factor_name}")

        if factor_name in self._cache:
            return self._cache[factor_name]

        df = factor.compute(data_bundle)
        factor.validate(df)
        self._cache[factor_name] = df
        return df

    def get_registry(self) -> FactorRegistry:
        """获取因子注册表。"""
        return self._registry

    def list_factors(self) -> list[str]:
        """列出所有可用因子。"""
        return self._registry.list_all()
