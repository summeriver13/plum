"""指数代码 → 中文名称映射（报告 / 图表展示用）。

从已删除的旧同步模块 ``app.data.syncer.coordinator`` 迁出，
只保留展示所需的名称映射。
"""

from __future__ import annotations

INDEX_NAMES: dict[str, str] = {
    # 宽基核心
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "000001.SH": "上证综指",
    # 风格 / 规模
    "000016.SH": "上证50",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "932000.SH": "中证2000",
    "399303.SZ": "国证2000",
    # 其它
    "000688.SH": "科创50",
    "000985.SH": "中证全指",
}


def index_name(symbol: str) -> str:
    """返回指数中文名；未知代码原样返回。

    兼容三种写法：``399303.SZ``（数据湖代码）、``IDX.399303.SZ``、
    ``IDX.399303``（plum 内部代码：``IDX.`` 前缀且不带交易所后缀）。
    """
    if not symbol:
        return ""
    code = symbol.upper().removeprefix("IDX.")
    if code in INDEX_NAMES:
        return INDEX_NAMES[code]
    bare = code.split(".", 1)[0]
    for key, name in INDEX_NAMES.items():
        if key.split(".", 1)[0] == bare:
            return name
    return symbol
