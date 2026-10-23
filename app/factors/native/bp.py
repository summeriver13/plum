"""BP 因子 — 市净率倒数。

Book-to-Price = 1 / PB_MRQ

值越高代表账面价值相对市值越高（越便宜）。
数据源: Bar.pb_mrq（来自 BaoStock / QMT 日线 K 线 API）。
"""

from __future__ import annotations

import pandas as pd

from app.factors.base.factor import DataBundle, Factor, FactorCategory


class NativeBP(Factor):
    """市净率倒数 (Book-to-Price)。

    name = "NATIVE_BP"
    category = VALUATION
    indicator = BP
    period = MRQ
    processing = RAW

    计算: value = 1.0 / pb_mrq
    """

    name = "NATIVE_BP"
    category = FactorCategory.VALUATION
    indicator = "BP"
    period = "MRQ"
    processing = "RAW"
    description = "市净率倒数 = 1 / PB_MRQ。值越高账面价值越便宜。"
    dependencies: list[str] = []
    data_source = "Plum_Bar"

    def compute(
        self,
        data_bundle: DataBundle,
        pit_context=None,
    ) -> pd.DataFrame:
        records: list[dict] = []
        for symbol, df in data_bundle._bars.items():
            if "pb_mrq" not in df.columns:
                continue
            pb = df["pb_mrq"].astype(float)
            bp = 1.0 / pb.replace(0, float("nan"))
            df = df.copy()
            df["value"] = bp
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
