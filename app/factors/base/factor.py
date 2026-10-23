"""因子基类 — ABC、注册表和通用类型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from app.config.paths import FINANCIAL_PATH

if TYPE_CHECKING:
    from app.data.financial.store import FinancialStore
    from app.factors.storage.pit import PITContext


class FactorCategory(Enum):
    """因子分类枚举。"""

    VALUATION = "Valuation"
    QUALITY = "Quality"
    MOMENTUM = "Momentum"
    GROWTH = "Growth"
    SIZE = "Size"
    VOLATILITY = "Volatility"
    LIQUIDITY = "Liquidity"
    SENTIMENT = "Sentiment"
    COMPOSITE = "Composite"


@dataclass
class DataBundle:
    """因子计算数据容器。

    封装因子 compute() 方法可访问的原始数据源。
    """

    _data: dict[str, pd.DataFrame] = field(default_factory=dict)
    _bars: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, key: str) -> pd.DataFrame | None:
        """通过键获取数据。"""
        return self._data.get(key)

    def get_bars(self, symbol: str) -> pd.DataFrame | None:
        """获取指定股票的 K 线数据。"""
        return self._bars.get(symbol)

    def put(self, key: str, df: pd.DataFrame) -> None:
        """存储数据。"""
        self._data[key] = df

    def put_bars(self, symbol: str, df: pd.DataFrame) -> None:
        """存储 K 线数据。"""
        self._bars[symbol] = df

    def keys(self) -> list[str]:
        """列出可用数据键。"""
        return list(self._data.keys())

    def attach_financials(
        self,
        store: FinancialStore | None = None,
        store_path: str = str(FINANCIAL_PATH),
    ) -> None:
        """将财务数据合并到日线 Bar 中。

        从 FinancialStore 读取预处理后的财务数据，
        PIT 对齐后合并到 self._bars 的每只股票 DataFrame 中。
        合并后因子即可从 df.columns 直接读取如 '归母净利润'、'ROE' 等列。

        Args:
            store: 已填充的 FinancialStore 实例。若为 None，从 store_path 创建。
            store_path: FinancialStore 根目录，默认 data/financial

        Raises:
            RuntimeError: 若无法创建 FinancialStore
        """
        if store is None:
            from app.data.financial.store import FinancialStore

            store = FinancialStore(root=store_path)

        from app.data.financial.integration import merge_financials_to_bars

        self._bars = merge_financials_to_bars(self._bars, store)


@dataclass
class FactorMetadata:
    """因子元数据。"""

    name: str = ""
    category: str = ""
    indicator: str = ""
    period: str = ""
    processing: str = ""
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    data_source: str = ""
    version: str = "1.0.0"
    last_updated: pd.Timestamp | None = None


class Factor(ABC):
    """所有因子的抽象基类。

    命名约定：Category_Indicator_Period_Processing
    示例：VAL_PE_TTM_ZSCORE

    生命周期：
    1. 注册表在导入时自动注册因子
    2. 管道实例化因子并调用 compute()
    3. 分层存储将输出持久化到适当的层
    """

    # 规范因子名称（如 "VAL_PE_TTM_ZSCORE"）
    name: ClassVar[str] = ""
    # 因子分类
    category: ClassVar[FactorCategory]
    # 指标类型（如 "PE"、"ROE"、"RET"）
    indicator: ClassVar[str] = ""
    # 周期说明（如 "TTM"、"LYR"、"1M"、"3M"、"1Y"）
    period: ClassVar[str] = ""
    # 处理方式（如 "ZSCORE"、"RANK"、"WINSORIZE"）
    processing: ClassVar[str] = ""
    # 描述
    description: ClassVar[str] = ""
    # 对其他因子的依赖（因子名称）
    dependencies: ClassVar[list[str]] = []
    # 数据源（如 "BaoStock"、"QMT"、"Wind"）
    data_source: ClassVar[str] = ""

    def __init__(self) -> None:
        self._validate_class_vars()

    def _validate_class_vars(self) -> None:
        """验证类变量是否正确设置。"""
        if not self.name:
            raise ValueError(f"{self.__class__.__name__}: 'name' 类变量必须设置")
        if not self.category:
            raise ValueError(f"{self.__class__.__name__}: 'category' 类变量必须设置")

    @abstractmethod
    def compute(
        self,
        data_bundle: DataBundle,
        pit_context: PITContext | None = None,
    ) -> pd.DataFrame:
        """计算因子值。

        参数:
            data_bundle: 包含所需输入数据的容器
            pit_context: 用于版本控制的点时上下文

        返回:
            具有 MultiIndex (timestamp, symbol) 和至少 'value' 列的 DataFrame。
            必须处理 PIT 对齐 — 仅使用每个时间戳可用的数据。
        """

    def validate(self, result: pd.DataFrame) -> None:
        """验证因子计算输出。"""
        if result.empty:
            return

        required_columns = {"value"}
        missing = required_columns - set(result.columns)
        if missing:
            raise ValueError(f"{self.name}: 缺少必需列: {missing}")

        if not isinstance(result.index, pd.MultiIndex):
            raise ValueError(f"{self.name}: 结果必须有 MultiIndex (timestamp, symbol)")

        if result.index.nlevels != 2:
            raise ValueError(f"{self.name}: 结果必须有 2 级 MultiIndex")

    def get_metadata(self) -> FactorMetadata:
        """获取因子元数据。"""
        return FactorMetadata(
            name=self.name,
            category=self.category.value,
            indicator=self.indicator,
            period=self.period,
            processing=self.processing,
            description=self.description,
            dependencies=self.dependencies,
            data_source=self.data_source,
        )


def canonical_name(
    category: str,
    indicator: str,
    period: str,
    processing: str,
) -> str:
    """从组件构建规范因子名称。

    返回:
        规范名称，如 "VAL_PE_TTM_ZSCORE"
    """
    return f"{category}_{indicator}_{period}_{processing}"
