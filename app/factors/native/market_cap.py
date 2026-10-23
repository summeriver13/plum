"""市值因子 — 总市值。

Market Cap = total_mv

小市值效应是 A 股最强的因子之一。
数据源: Bar.total_mv（来自 QMT 日线 K 线 API，BaoStock 不含此字段）。
"""

from __future__ import annotations

import pandas as pd

from app.factors.base.factor import DataBundle, Factor, FactorCategory


class NativeMarketCap(Factor):
    """总市值因子。

    name = "NATIVE_MCAP"
    category = SIZE
    indicator = MCAP
    period = DAILY
    processing = RAW

    计算: value = total_mv
    """

    name = "NATIVE_MCAP"
    category = FactorCategory.SIZE
    indicator = "MCAP"
    period = "DAILY"
    processing = "RAW"
    description = "总市值。小市值效应。"
    dependencies: list[str] = []
    data_source = "Plum_Bar"

    def compute(
        self,
        data_bundle: DataBundle,
        pit_context=None,
    ) -> pd.DataFrame:
        records: list[dict] = []
        for symbol, df in data_bundle._bars.items():
            if "total_mv" not in df.columns:
                continue
            mv = df["total_mv"].astype(float)
            df = df.copy()
            df["value"] = mv
            for ts_idx, row in df.iterrows():
                if isinstance(ts_idx, pd.Timestamp) and pd.notna(row["value"]):
                    records.append(
                        {
                            "timestamp": ts_idx,
                            "symbol": symbol,
                            "value": float(row["value"]),
                        }
                    )
        if not records:
            return pd.DataFrame(
                columns=["value"],
                index=pd.MultiIndex.from_tuples([], names=["timestamp", "symbol"]),
            )
        result = pd.DataFrame(records)
        return result.set_index(["timestamp", "symbol"])
