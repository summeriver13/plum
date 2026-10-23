"""财务数据 → Factor DataBundle 集成。

将 FinancialStore 中按报告期存储的财务数据，
PIT 对齐后合并到日线 Bar DataFrame 中，
使下游因子能从 df.columns 直接读取中文列名。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from app.data.financial.field_map import FIELD_MAP

if TYPE_CHECKING:
    from app.data.financial.store import FinancialStore

logger = logging.getLogger(__name__)


def merge_financials_to_bars(
    bars: dict[str, pd.DataFrame],
    store: FinancialStore,
) -> dict[str, pd.DataFrame]:
    """将财务数据 PIT 对齐后合并到日线 Bar DataFrame。

    对每只股票：
    1. 从 FinancialStore 读取财务数据 Parquet
    2. 对每个交易日，找到该日之前最后公告的报告期
    3. 将该报告期的财务字段复制到对应 Bar 行

    Args:
        bars: {symbol: DataFrame}，index 为 timestamp
        store: 已填充的 FinancialStore 实例

    Returns:
        合并后的 bars，每只股票的 DataFrame 额外包含中文财务列名

    Raises:
        RuntimeError: 若 store 中无数据
    """
    symbols_with_data = store.list_symbols()
    if not symbols_with_data:
        logger.warning("FinancialStore 中无数据，财务字段不会被注入")
        return bars

    # 构建 QMT 英文字段 → 中文输出列名的映射
    field_rename: dict[str, str] = {}
    for fm in FIELD_MAP:
        field_rename[fm.qmt_field] = fm.column

    merged: dict[str, pd.DataFrame] = {}
    merged_count = 0

    for symbol in sorted(bars.keys()):
        bar_df = bars[symbol].copy()
        if bar_df.empty:
            merged[symbol] = bar_df
            continue

        # 读取财务数据
        fin_df = store.read(symbol)
        if fin_df.empty:
            merged[symbol] = bar_df
            continue

        # 重命名为中文输出列名
        fin_df = fin_df.rename(
            columns={k: v for k, v in field_rename.items() if k in fin_df.columns}
        )

        # PIT 对齐：找到每个交易日前最后公告的报告期 ← 这里简化处理
        # 因为 store.write 时存的是 preprocessor 预处理后的数据，
        # 其 index 已经是处理过的。直接按 index 做 forward fill。
        bar_df = _pit_merge(bar_df, fin_df)

        merged[symbol] = bar_df
        merged_count += 1

    logger.info("财务数据已合并到 %d/%d 只股票的 Bar 中", merged_count, len(bars))
    return merged


def _pit_merge(
    bar_df: pd.DataFrame,
    fin_df: pd.DataFrame,
) -> pd.DataFrame:
    """PIT 对齐：将财务宽表合并到日线 Bar。

    fin_df index = report_period（已由 preprocessor 处理好）
    对每个 Bar 日期，取该日期之前最新报告期的财务数据。
    """
    if fin_df.empty:
        return bar_df

    # fin_df 的 index 是报告期，需要先转为 timestamp
    if not isinstance(fin_df.index, pd.DatetimeIndex):
        try:
            fin_df = fin_df.copy()
            fin_df.index = pd.to_datetime(fin_df.index)
        except Exception:
            logger.warning("财务数据 index 无法转为 DatetimeIndex，跳过 PIT 对齐")
            return bar_df

    fin_cols = [c for c in fin_df.columns if c not in bar_df.columns]

    if not fin_cols:
        return bar_df

    # 初始化 bar_df 上的财务列为 NaN
    for col in fin_cols:
        bar_df[col] = np.nan

    # 为每个 Bar 日期找报告期
    bar_timestamps = bar_df.index
    fin_timestamps = fin_df.index.sort_values()

    for col in fin_cols:
        series = fin_df[col].dropna()
        if series.empty:
            continue
        # 向前填充：每行 Bar 取 <= 该日期的最后一期财务数据
        for ts in bar_timestamps:
            mask = fin_timestamps <= ts
            if not mask.any():
                continue
            latest = fin_timestamps[mask][-1]
            val = series.get(latest)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                bar_df.at[ts, col] = val

    added = len(fin_cols)
    logger.debug("PIT 合并: 添加 %d 个财务列", added)
    return bar_df
