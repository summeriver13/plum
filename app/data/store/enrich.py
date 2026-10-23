"""一次性：用 QMT 给缓存里的 Bar 补 PE/PB/总市值。

QMT 的 get_market_data_ex 返回 PE/PB 列，Baostock 没有。
此脚本遍历所有已缓存的股票，从 QMT 拉 PE/PB，合并到 parquet。
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal

from app.data.connectors.qmt import _get_xtdata
from app.data.schema import Bar
from app.data.store.market import MarketCatalog

logger = logging.getLogger(__name__)


def enrich_bars_with_financials():
    catalog = MarketCatalog()
    u = catalog.get_universe()
    symbols = u.get("stocks", [])
    if not symbols:
        logger.info("No stocks in universe")
        return

    xtdata = _get_xtdata()
    total = len(symbols)
    enriched = 0
    skipped = 0

    for i, sym in enumerate(symbols):
        if i % 100 == 0:
            logger.info(
                "Progress: %d/%d (%d enriched, %d skipped)", i, total, enriched, skipped
            )

        bars = catalog.load_bars(sym, "1d")
        if not bars:
            skipped += 1
            continue

        # Check if PE/PB already populated
        has_pe = any(b.pe_ttm is not None and b.pe_ttm > 0 for b in bars[-10:])
        if has_pe:
            skipped += 1
            continue

        # Download PE/PB from QMT for the full date range
        start = bars[0].timestamp.strftime("%Y%m%d")
        end = bars[-1].timestamp.strftime("%Y%m%d")

        try:
            xtdata.download_history_data2(
                [sym],
                "1d",
                start,
                end,
                callback=lambda x: None,
            )
            data = xtdata.get_market_data_ex(
                field_list=[
                    "time",
                    "close",
                    "pe_ttm",
                    "pb_mrq",
                    "totalMarketCap",
                    "circulatingMarketCap",
                ],
                stock_list=[sym],
                period="1d",
                start_time=start,
                end_time=end,
            )

            if not data or sym not in data or data[sym].empty:
                skipped += 1
                continue

            df = data[sym]
            # Build date → PE/PB/市值 map
            pe_map = {}
            pb_map = {}
            mv_map = {}
            circ_mv_map = {}
            for _, row in df.iterrows():
                ts = row["time"]
                if isinstance(ts, (int, float)):
                    d = datetime.fromtimestamp(int(ts) // 1000).date()
                else:
                    d = ts.date() if hasattr(ts, "date") else ts
                pe = row.get("pe_ttm", 0)
                pb = row.get("pb_mrq", 0)
                mv = row.get("totalMarketCap", 0)
                cmv = row.get("circulatingMarketCap", 0)
                if pe and pe > 0:
                    pe_map[d] = Decimal(str(pe))
                if pb and pb > 0:
                    pb_map[d] = Decimal(str(pb))
                if mv and mv > 0:
                    mv_map[d] = Decimal(str(mv))
                if cmv and cmv > 0:
                    circ_mv_map[d] = Decimal(str(cmv))

            if not pe_map and not pb_map:
                skipped += 1
                continue

            # Merge into bars
            new_bars = []
            changed = False
            for b in bars:
                d = b.timestamp.date()
                new_pe = pe_map.get(d, b.pe_ttm)
                new_pb = pb_map.get(d, b.pb_mrq)
                new_mv = mv_map.get(d, getattr(b, "total_mv", None))
                new_cmv = circ_mv_map.get(d, getattr(b, "circ_mv", None))
                if new_pe != b.pe_ttm or new_pb != b.pb_mrq:
                    changed = True
                new_bars.append(
                    Bar(
                        symbol=b.symbol,
                        timestamp=b.timestamp,
                        open=b.open,
                        high=b.high,
                        low=b.low,
                        close=b.close,
                        volume=b.volume,
                        amount=b.amount,
                        pre_close=b.pre_close,
                        adj_factor=b.adj_factor,
                        is_st=b.is_st,
                        is_delisted=b.is_delisted,
                        pe_ttm=new_pe,
                        pb_mrq=new_pb,
                        ps_ttm=b.ps_ttm,
                        pcf_ncf_ttm=b.pcf_ncf_ttm,
                        total_mv=new_mv,
                        circ_mv=new_cmv,
                    )
                )

            if changed:
                catalog.save_bars(new_bars, sym, "1d")
                enriched += 1
            else:
                skipped += 1

        except Exception:
            skipped += 1

    logger.info("Done: %d enriched, %d skipped out of %d", enriched, skipped, total)


if __name__ == "__main__":
    enrich_bars_with_financials()
