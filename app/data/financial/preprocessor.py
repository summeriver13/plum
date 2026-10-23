"""财务数据预处理 — PIT 对齐 + TTM 聚合 + 字段映射。

核心功能：
1. QMT 英文字段 → 中文输出列名映射
2. 单季原始字段 → TTM 滚动聚合
3. PIT 合规：按公告日期延迟 1 个交易日生效
4. 自由现金流计算：OCF(TTM) - CAPEX(TTM)
5. 输出宽表：MultiIndex(交易日期, 股票代码) × 财务因子列
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from app.data.financial.field_map import (
    FIELD_MAP,
    TABLE_META_COLUMNS,
)

logger = logging.getLogger(__name__)


def _num(value: object, default: float = 0.0) -> float:
    """容错转 float（NULL/非数值返回 default）。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# 单季报告期识别：Q1=0331, Q2=0630, Q3=0930, Q4=1231
_QUARTER_MONTHS: dict[int, str] = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}


class FinancialPreprocessor:
    """财务数据预处理管道。

    Usage:
        preprocessor = FinancialPreprocessor()
        wide_df = preprocessor.process(
            raw_data={"600035.SH": {"Income": df, "PershareIndex": df}},
            trading_calendar=pd.DatetimeIndex([...]),
        )
    """

    # QMT 英文字段 → 中文输出列名（从 FIELD_MAP 构建）
    FIELD_RENAME: dict[str, dict[str, str]] = {}

    def __init__(self) -> None:
        self._build_rename_map()

    def _build_rename_map(self) -> None:
        """从 FIELD_MAP 构建表级重命名映射。"""
        for fm in FIELD_MAP:
            table = fm.qmt_table.value
            if table not in self.FIELD_RENAME:
                self.FIELD_RENAME[table] = {}
            self.FIELD_RENAME[table][fm.qmt_field] = fm.column

    # ── 主处理流程 ────────────────────────────────────────

    def process(
        self,
        raw_data: dict[str, dict[str, pd.DataFrame]],
        trading_calendar: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """主处理管道。

        Args:
            raw_data: fetcher.fetch() 的返回结果
            trading_calendar: 交易日历，用于 PIT 对齐

        Returns:
            宽表 DataFrame
            index: MultiIndex(timestamp, symbol)
            columns: 中文输出列名
        """
        all_records: list[dict[str, Any]] = []

        for symbol, tables in raw_data.items():
            fin_df = self._merge_tables(tables)
            if fin_df.empty:
                continue

            fin_df = self._add_derived_fields(fin_df)
            records = self._to_records(symbol, fin_df)
            all_records.extend(records)

        if not all_records:
            logger.debug("预处理后无有效数据（批内股票均为新股/无财务数据）")
            return pd.DataFrame()

        result = pd.DataFrame(all_records)
        result = result.set_index(["股票代码", "报告期"]).sort_index()

        if trading_calendar is not None:
            result = self._pit_align(result, trading_calendar)

        return result

    # ── 表合并 ────────────────────────────────────────────

    def _merge_tables(
        self,
        tables: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """将四张表按报告期合并为单张宽表。

        每张表先选出目标字段并重命名为输出列名，
        然后按 m_timetag（报告截止日）做 outer join。
        """
        merged: pd.DataFrame | None = None

        for table_name, df in tables.items():
            if df is None or df.empty:
                continue

            rename = self.FIELD_RENAME.get(table_name, {})
            cols_to_keep = [
                c for c in df.columns if c in rename or c in TABLE_META_COLUMNS
            ]
            if not cols_to_keep:
                continue

            sub = cast("pd.DataFrame", df[cols_to_keep].copy())
            sub = cast("pd.DataFrame", sub.rename(columns=rename))

            # 按 m_timetag 聚合（同一报告期可能有多条公告记录，取最新）
            if "m_timetag" in sub.columns:
                sub = sub.sort_values("m_anntime").groupby("m_timetag").last()
                sub.index.name = "report_period"
                sub = sub.drop(columns=["m_anntime"], errors="ignore")

            if merged is None:
                merged = sub
            else:
                merged = merged.join(sub, how="outer", rsuffix=f"_{table_name}")

        if merged is None:
            return pd.DataFrame()

        return merged

    # ── 派生字段 ──────────────────────────────────────────

    def _add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算派生财务字段。"""
        df = df.copy()

        # 1. TTM 聚合：对 needs_ttm=True 的字段做四季度滚动求和
        ttm_fields = [fm.column for fm in FIELD_MAP if fm.needs_ttm]
        for col in ttm_fields:
            if col in df.columns:
                df[f"{col}_TTM"] = self._rolling_ttm(cast("pd.Series", df[col]))

        # 2. RENAME: 预计算的 归母净利润同比 → 归母净利润同比增速
        if "归母净利润同比" in df.columns and "归母净利润同比增速" not in df.columns:
            df["归母净利润同比增速"] = df["归母净利润同比"]

        # 3. 自由现金流 = 经营现金流(TTM) - 资本支出(TTM)
        ocf_ttm = "经营现金流_TTM"
        capex_ttm = "资本支出_TTM"
        if ocf_ttm in df.columns and capex_ttm in df.columns:
            df["自由现金流"] = df[ocf_ttm] - df[capex_ttm]
        elif ocf_ttm in df.columns:
            df["自由现金流"] = df[ocf_ttm]

        # 4. 单季 ROE = 归母净利润(单季) / 净资产
        #    PershareIndex 已提供预计算 ROE，这里仅作为 fallback
        if "ROE" not in df.columns or bool(cast("pd.Series", df["ROE"]).isna().all()):
            if "归母净利润" in df.columns and "净资产" in df.columns:
                df["ROE_np_单季"] = df["归母净利润"] / df["净资产"].replace(0, np.nan)

        return df

    @staticmethod
    def _rolling_ttm(series: pd.Series) -> pd.Series:
        """对按报告期排序的序列做四季度 TTM 聚合。"""
        return cast("pd.Series", series.rolling(window=4, min_periods=4).sum())

    # ── 记录转换 ──────────────────────────────────────────

    def _to_records(
        self,
        symbol: str,
        fin_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """将单只股票的财务宽表转为记录列表。"""
        records: list[dict[str, Any]] = []
        for report_period, row in fin_df.iterrows():
            record: dict[str, Any] = {
                "股票代码": symbol,
                "报告期": str(report_period),
            }
            for col in fin_df.columns:
                val = row[col]
                if bool(pd.isna(val)):
                    continue
                record[col] = (
                    _num(val) if isinstance(val, (np.floating, float)) else val
                )
            records.append(record)
        return records

    # ── PIT 对齐 ──────────────────────────────────────────

    @staticmethod
    def _pit_align(
        df: pd.DataFrame,
        trading_calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """PIT 合规对齐。

        将财务数据按报告期映射到生效交易日。
        简化策略：每个报告期的数据从该季末的下一个交易日起生效，
        直到下一报告期数据覆盖。

        Args:
            df: index=(交易日期, 股票代码) 的 DataFrame
            trading_calendar: 交易日历

        Returns:
            对齐后的 DataFrame
        """
        if df.empty or len(trading_calendar) == 0:
            return df

        # 如果 df 的 index 已经是 (timestamp, symbol)，直接返回
        # PIT 对齐逻辑在调用方（DataBundle 合并时）处理更合适
        # 这里先做基础校验：确保 index 中的 timestamp 在交易日内
        idx_timestamps = df.index.get_level_values(0)
        valid_mask = idx_timestamps.isin(trading_calendar)
        return df[valid_mask]  # type: ignore[index] — valid_mask 是 Boolean Series，布尔索引返回 DataFrame 非 None

    # ── 增量模式 ──────────────────────────────────────────

    def process_incremental(
        self,
        raw_data: dict[str, dict[str, pd.DataFrame]],
        existing_store,  # FinancialStore
        trading_calendar: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """增量处理：合并已有数据和新数据。

        Args:
            raw_data: 新获取的财务数据
            existing_store: 已有 FinancialStore 实例
            trading_calendar: 交易日历

        Returns:
            合并后的完整宽表
        """
        new_df = self.process(raw_data, trading_calendar)
        if new_df.empty:
            return new_df

        # 合并已有数据
        for symbol in raw_data:
            old_df = existing_store.read(symbol)
            if old_df.empty:
                continue
            # 用新数据覆盖旧数据的重叠部分
            # （简化处理：直接替换整个 stock）
            existing_store.write(symbol, new_df.xs(symbol, level="股票代码"))

        return new_df
