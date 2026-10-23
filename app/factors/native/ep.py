"""EP 因子 — 盈利收益率。

Earnings Yield = 1 / PE_TTM

值越高代表估值越低（越便宜），是价值因子的核心。
数据源: Bar.pe_ttm（来自 BaoStock / QMT 日线 K 线 API）。
"""

from __future__ import annotations

import pandas as pd

from app.factors.base.factor import DataBundle, Factor, FactorCategory


class NativeEP(Factor):
    """盈利收益率 (Earnings Yield)。

    name = "NATIVE_EP"
    category = VALUATION
    indicator = EP
    period = TTM
    processing = RAW

    计算: value = 1.0 / pe_ttm
    """

    name = "NATIVE_EP"
    category = FactorCategory.VALUATION
    indicator = "EP"
    period = "TTM"
    processing = "RAW"
    description = "盈利收益率 = 1 / PE_TTM。值越高估值越低。"
    dependencies: list[str] = []
    data_source = "Plum_Bar"

    def compute(
        self,
        data_bundle: DataBundle,
        pit_context=None,
    ) -> pd.DataFrame:
        records: list[dict] = []
        for symbol, df in data_bundle._bars.items():
            if "pe_ttm" not in df.columns:
                continue
            pe = df["pe_ttm"].astype(float)
            ep = 1.0 / pe.replace(0, float("nan"))
            df = df.copy()
            df["value"] = ep
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
