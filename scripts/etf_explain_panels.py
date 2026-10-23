"""ETF 轮动可解释性：行情面板加载与标的分类。

两个口径约定（与策略/引擎保持一致）：
- 后复权价 = close × adj_factor（动量与贡献都用它）；
- 分类先按代码前缀识别非 A 股资产（债/黄金/商品/跨境），其余按与 A 股宽基的相关性。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _num(value: Any, default: float = 0.0) -> float:
    """安全转 float（None/NaN/异常值回退默认值）。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def _int(value: Any, default: int = 0) -> int:
    """安全转 int（None/NaN/越界回退默认值）。"""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


PANEL_COLUMNS = ("close", "adj_factor", "amount")
ADJ_FACTOR_EPSILON = 1e-6
"""与引擎同阈：复权因子比值偏差低于该值时视为无公司行为。"""

# 解释用分类：先按代码前缀识别非 A 股资产，其余按与 A 股宽基的相关性判定
CODE_CLASSES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("511", "159001", "159005", "159008"), "债/货币"),
    (("518", "159934", "159937"), "黄金"),
    (("159985", "159980", "162411", "159981"), "商品"),
    (("513", "159920", "159941", "159866", "159892"), "跨境"),
)
A_SHARE_CORR = 0.6
"""与 A 股宽基日收益相关性达到该值即归为「A 股股票型」。"""


def load_panels(
    market_root: Path,
    symbols: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """读取 ETF 行情副本，返回 close / adj_factor / amount 宽表。"""
    frames: dict[str, dict[str, pd.Series]] = {column: {} for column in PANEL_COLUMNS}
    for exchange in ("SH", "SZ"):
        for path in sorted((market_root / "1d" / exchange).glob("*.parquet")):
            symbol = f"{exchange}.{path.stem}"
            if symbols is not None and symbol not in symbols:
                continue
            try:
                frame = pd.read_parquet(path, columns=list(PANEL_COLUMNS))
            except Exception:  # noqa: BLE001 - 单文件损坏不应中断分析
                logger.warning("行情读取失败：%s", path, exc_info=True)
                continue
            for column in PANEL_COLUMNS:
                series = frame[column]
                if isinstance(series, pd.DataFrame):  # 列重名时退化为 DataFrame，跳过
                    logger.warning("列 %s 在 %s 中不唯一，跳过", column, path)
                    continue
                frames[column][symbol] = series.astype("float64")
    return {column: pd.DataFrame(values) for column, values in frames.items()}


def adjusted_close(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """后复权收盘价（close × adj_factor）。"""
    return panels["close"] * panels["adj_factor"]


def classify_symbols(
    adj_close: pd.DataFrame,
    ref_symbols: tuple[str, ...] = ("SH.510300", "SH.510500", "SH.512100"),
) -> dict[str, str]:
    """给标的分类：代码前缀识别非股票资产，其余按与 A 股宽基的相关性。"""
    refs = [symbol for symbol in ref_symbols if symbol in adj_close.columns]
    ref_return = adj_close[refs].pct_change().mean(axis=1) if refs else None
    corr = (
        adj_close.pct_change().corrwith(ref_return) if ref_return is not None else None
    )

    result: dict[str, str] = {}
    for symbol in adj_close.columns:
        code = symbol.split(".")[-1]
        for prefixes, label in CODE_CLASSES:
            if code.startswith(prefixes):
                result[symbol] = label
                break
        else:
            value = _num(corr.get(symbol, 0.0)) if corr is not None else 0.0
            result[symbol] = "A股股票型" if value >= A_SHARE_CORR else "其他/低相关"
    return result
