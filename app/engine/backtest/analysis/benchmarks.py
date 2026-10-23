"""基准指数解析 —— 交易所指数（``IDX.*``）与 plum 自建风格指数（``STYLE.*``）。

策略的真实暴露是「等权微盘」风格：市值加权宽基（沪深300 / 中证1000）会把风格
收益整体算成 alpha。报告主基准因此改用自建风格指数：

    STYLE.MICROCAP    微盘等权风格指数（最小 2% 市值、日频等权、零成本）

自建指数由 ``scripts/build_microcap_index.py --publish`` 发布到 ``data/benchmarks/``。
不放进 ``data/market/``：后者是数据湖的派生副本，会被重建脚本整体覆盖，
混入合成序列会让基准静默消失（基准加载失败只降级为「无基准」，不会报错）。

用法::

    from app.engine.backtest.analysis.benchmarks import (
        benchmark_label,
        load_aligned_benchmark,
    )

    values = load_aligned_benchmark("STYLE.MICROCAP", timestamps)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config.paths import BENCHMARKS_PATH

logger = logging.getLogger(__name__)

STYLE_BENCHMARKS: dict[str, StyleBenchmark] = {}


def _safe_float(value: object) -> float | None:
    """容错转 float（numpy 标量 / Decimal / 字符串均可）；不可转时返回 None。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_timestamp(value: object) -> pd.Timestamp:
    """把任意时间表示归一化为 Timestamp；非法输入抛 ValueError。"""
    try:
        stamp = pd.Timestamp(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法解析时间戳: {value!r}") from exc
    if pd.isna(stamp):
        raise ValueError(f"非法时间戳: {value!r}")
    return stamp


def _as_date(value: object):
    """把时间戳归一化为日期；无法解析时返回 None。"""
    if isinstance(value, datetime):
        return value.date()
    try:
        return _as_timestamp(value).date()
    except ValueError:
        return None


@dataclass(frozen=True)
class StyleBenchmark:
    """plum 自建风格指数（发布产物 = data/benchmarks/<file>）。"""

    key: str
    label: str
    file: str
    note: str


STYLE_BENCHMARKS["STYLE.MICROCAP"] = StyleBenchmark(
    key="STYLE.MICROCAP",
    label="微盘等权风格指数",
    file="microcap_equal_weight.csv",
    note="最小 2% 市值等权、日频再平衡、零成本（T+1 选样）",
)

STYLE_BENCHMARKS["STYLE.QUALITY"] = StyleBenchmark(
    key="STYLE.QUALITY",
    label="质量等权风格指数",
    file="quality_equal_weight.csv",
    note="质量前 50 只等权、月度换仓、零成本（PIT 按法定披露期限对齐）",
)


def is_style_benchmark(symbol: str) -> bool:
    """是否为 plum 自建风格指数代码。"""
    return symbol.upper() in STYLE_BENCHMARKS


def benchmark_label(symbol: str) -> str:
    """基准显示名；无法识别的代码原样返回。"""
    if not symbol:
        return ""
    spec = STYLE_BENCHMARKS.get(symbol.upper())
    if spec is not None:
        return spec.label
    try:
        from app.data.index_names import index_name

        return index_name(symbol)
    except Exception:  # noqa: BLE001 — 名称只影响展示，失败不应中断报告
        logger.debug("指数名称解析失败: %s", symbol, exc_info=True)
        return symbol


def load_benchmark_prices(symbol: str) -> pd.Series | None:
    """加载基准收盘价序列（索引为交易日）。

    Returns:
        价格序列；代码未知或文件缺失时返回 None。
    """
    spec = STYLE_BENCHMARKS.get(symbol.upper())
    if spec is not None:
        return _load_style_prices(spec)
    return _load_index_prices(symbol)


def load_aligned_benchmark(
    symbol: str, timestamps: Sequence[datetime]
) -> list[float] | None:
    """加载基准价格并对齐到给定时间戳序列。

    任一时间戳在基准中缺失即返回 None（宁可不显示基准，也不做前向填充，
    否则会把非交易日/停牌日错配成 0 收益）。
    """
    prices = load_benchmark_prices(symbol)
    if prices is None or prices.empty:
        return None
    price_map: dict[object, float] = {}
    for ts, raw in prices.items():
        day = _as_date(ts)
        value = _safe_float(raw)
        if day is not None and value is not None:
            price_map[day] = value
    values: list[float] = []
    for point in timestamps:
        day = _as_date(point)
        value = price_map.get(day) if day is not None else None
        if value is None or not math.isfinite(value):
            logger.debug("基准 %s 缺少 %s 的数据，放弃对齐", symbol, day)
            return None
        values.append(value)
    return values


def publish_style_benchmark(key: str, prices: pd.Series) -> Path:
    """把风格指数净值序列发布成可加载的基准文件。

    Args:
        key: 风格指数代码（如 ``STYLE.MICROCAP``）
        prices: 净值/价格序列，索引为交易日

    Returns:
        发布文件路径
    """
    spec = STYLE_BENCHMARKS.get(key.upper())
    if spec is None:
        raise KeyError(f"未注册的风格指数: {key}")
    frame = pd.DataFrame(
        {
            "date": [str(_as_date(ts) or "") for ts in prices.index],
            "close": [_safe_float(v) or 0.0 for v in list(prices)],
        }
    )
    BENCHMARKS_PATH.mkdir(parents=True, exist_ok=True)
    target = BENCHMARKS_PATH / spec.file
    frame.to_csv(target, index=False, encoding="utf-8")
    logger.info("风格基准已发布: %s（%d 行）", target, len(frame))
    return target


def _load_style_prices(spec: StyleBenchmark) -> pd.Series | None:
    """读取自建风格指数（先 parquet 后 csv）。"""
    for name in (Path(spec.file).with_suffix(".parquet"), Path(spec.file)):
        target = BENCHMARKS_PATH / name
        if not target.exists():
            continue
        try:
            if target.suffix == ".parquet":
                frame = pd.read_parquet(target)
                if isinstance(frame.index, pd.DatetimeIndex):
                    frame = frame.reset_index()
                day_col = "date" if "date" in frame.columns else frame.columns[0]
            else:
                frame = pd.read_csv(target)
                day_col = "date" if "date" in frame.columns else frame.columns[0]
            series = pd.Series(
                pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float),
                index=pd.to_datetime(frame[day_col]),
            )
            return series.dropna()
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("风格基准读取失败: %s（%s）", target, exc)
            return None
    logger.warning(
        "风格基准文件缺失: %s —— 先运行 scripts/build_microcap_index.py --publish",
        BENCHMARKS_PATH / spec.file,
    )
    return None


def _load_index_prices(symbol: str) -> pd.Series | None:
    """从 MarketCatalog 读取交易所指数收盘价。"""
    try:
        from app.data.store.market import MarketCatalog

        bars = MarketCatalog().load_bars(symbol, "1d")
    except Exception:  # noqa: BLE001 — 基准缺失不应中断报告生成
        logger.warning("基准指数加载失败: %s", symbol, exc_info=True)
        return None
    if not bars:
        logger.warning("基准指数无数据: %s", symbol)
        return None
    closes: dict[pd.Timestamp, float] = {}
    for bar in bars:
        value = _safe_float(bar.close)
        if value is None:
            continue
        closes[pd.Timestamp(bar.timestamp)] = value
    return pd.Series(closes).sort_index()
