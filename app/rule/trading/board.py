"""A股板块配置与识别。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple


class Board(Enum):
    """A股板块类型。

    - MAIN: 主板（上交所/深交所主板）
    - GEM: 创业板（深交所）
    - STAR: 科创板（上交所）
    - BSE: 北交所
    """

    MAIN = "main"
    GEM = "gem"
    STAR = "star"
    BSE = "bse"


class PriceLimit(NamedTuple):
    """有效价格范围。

    属性:
        low: 最低价格（int ×10⁴）
        high: 最高价格（int ×10⁴）
    """

    low: int
    high: int


@dataclass(frozen=True)
class BoardConfig:
    """板块配置。

    属性:
        lot_size: 整手股数（买入时必须是整手）
        min_order: 最小委托股数
        max_limit_order: 限价单笔最大委托股数
        max_market_order: 市价单笔最大委托股数
        supports_odd_lot: 是否支持零股（买入时可在整手基础上加零股）
        price_limit_pct: 涨跌停幅度
    """

    lot_size: int
    min_order: int
    max_limit_order: int
    max_market_order: int
    supports_odd_lot: bool = False
    price_limit_pct: float = 0.1


# A股各板块配置
BOARD_CONFIGS: dict[Board, BoardConfig] = {
    Board.MAIN: BoardConfig(
        lot_size=100,
        min_order=100,
        max_limit_order=1_000_000,
        max_market_order=50_000,
        supports_odd_lot=False,
        price_limit_pct=0.1,
    ),
    Board.GEM: BoardConfig(
        lot_size=100,
        min_order=100,
        max_limit_order=1_000_000,
        max_market_order=50_000,
        supports_odd_lot=False,
        price_limit_pct=0.2,
    ),
    Board.STAR: BoardConfig(
        lot_size=200,
        min_order=200,
        max_limit_order=100_000,
        max_market_order=50_000,
        supports_odd_lot=True,
        price_limit_pct=0.2,
    ),
    Board.BSE: BoardConfig(
        lot_size=100,
        min_order=100,
        max_limit_order=1_000_000,
        max_market_order=50_000,
        supports_odd_lot=True,
        price_limit_pct=0.3,
    ),
}


def get_board(symbol: str) -> Board:
    """根据股票代码判断所属板块。

    参数:
        symbol: 股票代码（如 "600000.SH"、"000001.SZ"）

    返回:
        Board 板块枚举值
    """
    if "." not in symbol:
        return _infer_board_from_code(symbol)
    code, _ = symbol.split(".", 1)
    return _infer_board_from_code(code)


def _infer_board_from_code(code: str) -> Board:
    """根据股票代码号段推断板块。

    代码规则：
    - 688xxx/689xxx: 科创板
    - 430xxx/830xxx: 北交所
    - 300xxx/301xxx: 创业板
    - 其他: 主板
    """
    if not code.isdigit():
        return Board.MAIN

    n = int(code)

    if 688000 <= n <= 689999:
        return Board.STAR

    if 430000 <= n <= 499999 or 830000 <= n <= 899999:
        return Board.BSE

    if 300000 <= n <= 301999:
        return Board.GEM

    return Board.MAIN


def get_board_config(symbol: str) -> BoardConfig:
    """获取股票所属板块的交易配置。

    参数:
        symbol: 股票代码

    返回:
        BoardConfig 板块配置
    """
    return BOARD_CONFIGS[get_board(symbol)]
