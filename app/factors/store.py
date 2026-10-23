"""因子存储 — 矩阵存储与 O(1) 查找。"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock

import pandas as pd

from app.config.paths import FACTORS_PATH
from app.factors.base.registry import FactorDAG, FactorRegistry

logger = logging.getLogger(__name__)


@dataclass
class FactorMatrix:
    """时间 × 股票因子矩阵。

    存储为 DataFrame，时间戳为索引，股票为列。
    支持按时间戳 O(1) 查找。
    """

    name: str
    data: pd.DataFrame
    universe: set[str] = field(default_factory=set)

    def score_at(self, ts: datetime) -> dict[str, float]:
        """O(1) 查找指定时间戳的因子值。

        返回:
            {股票代码: 因子值} 字典，排除 NaN 值
        """
        if ts not in self.data.index:
            return {}

        row = self.data.loc[ts]
        result = row.dropna().to_dict()

        if self.universe:
            result = {s: v for s, v in result.items() if s in self.universe}

        return result

    def ranked_at(
        self,
        ts: datetime,
        top_n: int | None = None,
        ascending: bool = False,
    ) -> list[tuple[str, float]]:
        """获取指定时间戳的排序因子值。

        返回:
            [(股票代码, 因子值)] 列表，按因子值排序
        """
        scores = self.score_at(ts)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)

        if top_n is not None:
            ranked = ranked[:top_n]

        return ranked

    def save(self, path: Path) -> None:
        """保存因子矩阵到 Parquet。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_parquet(path, compression="snappy")

    @classmethod
    def load(
        cls, name: str, path: Path, universe: set[str] | None = None
    ) -> FactorMatrix:
        """从 Parquet 加载因子矩阵。"""
        data = pd.read_parquet(path)
        return cls(name=name, data=data, universe=universe or set())


class FactorStore:
    """存储和管理多个因子矩阵。

    提供多因子策略的 O(1) 查找接口。
    """

    def __init__(self, cache_dir: Path | str = str(FACTORS_PATH)) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._matrices: dict[str, FactorMatrix] = {}

    def register(self, matrix: FactorMatrix) -> None:
        """注册因子矩阵。"""
        self._matrices[matrix.name] = matrix

    def get(self, name: str) -> FactorMatrix | None:
        """按名称获取因子矩阵。"""
        return self._matrices.get(name)

    def score_at(
        self,
        ts: datetime,
        factor_names: list[str] | None = None,
    ) -> dict[str, float]:
        """获取指定时间戳的综合因子得分。

        如果多个因子，返回加权平均（等权重）。
        """
        names = factor_names or list(self._matrices.keys())

        if len(names) == 1:
            matrix = self._matrices[names[0]]
            return matrix.score_at(ts)

        scores_list = []
        for name in names:
            matrix = self._matrices.get(name)
            if matrix:
                scores = matrix.score_at(ts)
                if scores:
                    scores_list.append(pd.Series(scores))

        if not scores_list:
            return {}

        combined = pd.concat(scores_list, axis=1).mean(axis=1)
        return combined.dropna().to_dict()

    def ranked_at(
        self,
        ts: datetime,
        factor_names: list[str] | None = None,
        top_n: int | None = None,
        ascending: bool = False,
    ) -> list[tuple[str, float]]:
        """获取指定时间戳按因子得分排序的股票。"""
        scores = self.score_at(ts, factor_names)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)

        if top_n is not None:
            ranked = ranked[:top_n]

        return ranked

    def save_all(self) -> None:
        """保存所有矩阵到缓存目录。"""
        for name, matrix in self._matrices.items():
            path = self.cache_dir / f"{name}.parquet"
            matrix.save(path)

    def load_all(self, universe: set[str] | None = None) -> None:
        """从缓存目录加载所有矩阵。"""
        for path in self.cache_dir.glob("*.parquet"):
            name = path.stem
            matrix = FactorMatrix.load(name=name, path=path, universe=universe)
            self._matrices[name] = matrix


class CachingFactorStore:
    """带 L1/L2 分层缓存和 DAG 自动计算的因子存储。

    架构:
    - L1: 内存 LRU (OrderedDict)，最大 20 天
    - L2: 磁盘 Parquet 通过 TieredStorage
    - DAG: FactorDAG 用于依赖解析

    用法:
        store = CachingFactorStore(registry=None)
        store.register_computer("VAL_PE_TTM", my_factor_instance)
        store.compute_and_cache(["VAL_PE_TTM", "QUAL_ROE_TTM"], start, end)
        scores = store.score_at(ts, ["VAL_PE_TTM"])
    """

    def __init__(
        self,
        registry: FactorRegistry | None = None,
        cache_dir: str = str(FACTORS_PATH),
        l1_max_age_days: int = 20,
    ) -> None:
        """初始化因子存储。

        Note:
            L2（磁盘分层）尚未实现（原依赖的 ``app.factors.storage.tiered_storage``
            不存在），当前只有 L1 内存分层缓存。``cache_dir`` 为预留参数。
        """
        self._registry = registry or FactorRegistry.get_instance()
        self._dag = FactorDAG(self._registry)

        self._l1_cache: dict[str, OrderedDict[datetime, pd.DataFrame]] = {}
        self._l1_max_age = timedelta(days=l1_max_age_days)
        self._l1_lock = RLock()

        self._computers: dict[str, Callable[..., pd.DataFrame]] = {}
        self._matrices: dict[str, FactorMatrix] = {}

    def register_computer(
        self, factor_name: str, computer: Callable[..., pd.DataFrame]
    ) -> None:
        self._computers[factor_name] = computer

    def _get_l1(self, factor_name: str) -> OrderedDict[datetime, pd.DataFrame]:
        if factor_name not in self._l1_cache:
            self._l1_cache[factor_name] = OrderedDict()
        return self._l1_cache[factor_name]

    def _evict_l1_expired(self, factor_name: str) -> None:
        cutoff = datetime.now() - self._l1_max_age
        cache = self._get_l1(factor_name)
        expired = [k for k in cache if k < cutoff]
        for k in expired:
            del cache[k]

    def get(
        self,
        factor_name: str,
        ts: datetime,
    ) -> dict[str, float]:
        cache = self._get_l1(factor_name)
        if ts in cache:
            cache.move_to_end(ts)
            df = cache[ts]
        else:
            # L1 未命中：L2 未实现，直接视为无缓存
            return {}
        return self._df_to_scores(df, ts)

    def _df_to_scores(self, df: pd.DataFrame, ts: datetime) -> dict[str, float]:
        if ts not in df.index:
            return {}
        row = df.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[:, 0]
        return row.dropna().to_dict()  # type: ignore[return-value] — dropna().to_dict() 返回 dict 但 pandas 类型标注为 Unknown

    def compute_and_cache(
        self,
        factor_names: list[str],
        start: datetime,
        end: datetime,
        universe: set[str] | None = None,
    ) -> None:
        ordered = self._dag.get_computation_order(factor_names)

        for fname in ordered:
            if fname not in self._computers:
                continue
            computer = self._computers[fname]
            bars = self._gather_dependency_bars(fname, start, end, universe)
            result = computer(bars)
            if not isinstance(result, pd.DataFrame):
                continue
            result_df = pd.DataFrame(result)
            if universe:
                result_df = result_df[[c for c in result_df.columns if c in universe]]
            with self._l1_lock:
                cache = self._get_l1(fname)
                for ts_val in result_df.index:
                    if isinstance(ts_val, pd.Timestamp):
                        tval = ts_val.to_pydatetime()
                    else:
                        tval = ts_val
                    cache[tval] = pd.DataFrame(result_df.loc[[tval]])  # type: ignore[index] — pandas OrderedDict.__setitem__ 键类型推断过窄
                self._evict_l1_expired(fname)
            matrix = FactorMatrix(
                name=fname,
                data=pd.DataFrame(result_df),
                universe=universe or set(),
            )
            self._matrices[fname] = matrix

    def _gather_dependency_bars(
        self,
        factor_name: str,
        start: datetime,
        end: datetime,
        universe: set[str] | None = None,
    ) -> pd.DataFrame:
        # L2（磁盘分层）未实现，无历史缓存可回补，故返回空表。
        # 参数签名保持不变，便于后续实现 L2 时直接接入。
        return pd.DataFrame()

    def score_at(
        self,
        ts: datetime,
        factor_names: list[str] | None = None,
    ) -> dict[str, float]:
        names = factor_names or list(self._matrices.keys())
        if len(names) == 1:
            return self.get(names[0], ts)
        scores_list: list[pd.Series] = []
        for name in names:
            scores = self.get(name, ts)
            if scores:
                scores_list.append(pd.Series(scores))
        if not scores_list:
            return {}
        combined = pd.concat(scores_list, axis=1).mean(axis=1)
        return combined.dropna().to_dict()  # type: ignore[return-value] — dropna().to_dict() 返回 dict 但 pandas 类型标注为 Unknown

    def ranked_at(
        self,
        ts: datetime,
        factor_names: list[str] | None = None,
        top_n: int | None = None,
        ascending: bool = False,
    ) -> list[tuple[str, float]]:
        scores = self.score_at(ts, factor_names)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
        if top_n is not None:
            ranked = ranked[:top_n]
        return ranked

    def invalidate(self, factor_name: str) -> None:
        self._l1_cache.pop(factor_name, None)

    def l1_stats(self) -> dict[str, object]:
        total_entries = sum(len(v) for v in self._l1_cache.values())
        factors = len(self._l1_cache)
        return {"total_entries": total_entries, "factors_cached": factors}
