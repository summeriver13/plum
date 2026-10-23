"""行业 ETF 池构建：按名称映射行业 + 同质去重（供诊断脚本与策略复用）。

口径：
- 只留 A 股股票型（与非 ETF 无关：按与 A 股宽基相关性判定，见 etf_explain_panels）；
- 先排除宽基/风格/跨境/商品/债券类，再按「具体度优先」匹配行业；
- 同行业只留近期成交额最大的 1 只，跨行业按 60 日收益相关性 > 0.95 去重。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

PLUM_ROOT = Path(__file__).resolve().parents[1]
if str(PLUM_ROOT) not in sys.path:  # 允许直接导入本模块时解析 app.*
    sys.path.insert(0, str(PLUM_ROOT))

from app.data.lake import dataset_dir  # noqa: E402
from scripts.etf_explain_panels import _num  # noqa: E402

INSTRUMENTS_DATASET = "instruments"
INSTRUMENTS_FILE = "part-0001.parquet"
MIN_BARS = 400
MIN_AMOUNT = 1.0e7
CORR_DEDUP = 0.95

EXCLUDE_KEYWORDS = (
    "港股|恒生|H股|纳斯达克|标普|道琼斯|日经|德国|法国|东南亚|越南|沙特|黄金|白银|"
    "原油|商品|豆粕|国债|政金债|信用债|债券|货币|短融|同业存单|可转债|REIT|ESG|"
    "MSCI|红利|低波|等权|价值|成长|基本面|质量|动量|龙头|央企|国企|中央企业|"
    "300|500|1000|2000|50|100|180|创业板|科创|深证|上证|中证A|全指|双创|A50|A500|"
    "行业|主题|策略|现金流|自由"
)
"""排除宽基/风格/跨境/商品/债券类，只留行业与板块 ETF。"""

INDUSTRY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("白酒", "白酒"),
    ("食品饮料", "食品|饮料|乳业"),
    ("医药", "医药|医疗|生物|疫苗|中药|创新药|药"),
    ("半导体", "半导体|芯片|集成电路"),
    ("军工", "军工|国防|航天|航空"),
    ("新能源", "新能源|光伏|电池|锂|储能|风电|核电|氢能"),
    ("有色", "有色|稀土|稀有金属"),
    ("煤炭", "煤炭"),
    ("钢铁", "钢铁|特钢"),
    ("地产", "地产|房地产|不动产"),
    ("建材", "建材|基建|建筑"),
    ("汽车", "汽车|整车"),
    ("机械", "机械|机床|机器人"),
    ("农业", "农业|养殖|畜牧|种业|种子"),
    ("传媒", "传媒|游戏|影视|动漫|出版"),
    ("计算机", "计算机|软件|云计算|大数据|人工智能|信创|数字经济"),
    ("通信", "通信|5G|光通信"),
    ("电子", "电子|面板"),
    ("化工", "化工|化学|新材料|材料"),
    ("环保", "环保|碳中和|污水|低碳"),
    ("电力", "电力|公用|电网|绿电"),
    ("交运", "交运|物流|航运|港口|高速"),
    ("银行", "银行"),
    ("证券", "证券|券商|非银"),
    ("保险", "保险"),
    ("家电", "家电"),
    ("旅游", "旅游|餐饮|酒店|免税"),
    ("科技", "科技|创新|智能制造|高端装备"),
    ("消费", "消费|零售|商贸"),
)

EXCLUDE_RE = re.compile(EXCLUDE_KEYWORDS)
INDUSTRY_RES = tuple((name, re.compile(pattern)) for name, pattern in INDUSTRY_KEYWORDS)


def load_names(instruments: Path | None = None) -> pd.Series:
    """读取标的名称表（symbol → name），代码统一为 ``交易所.代码``。

    路径经 ``app.data.lake`` 解析（QDATA_ROOT / market-data/.env），不硬编码湖路径。
    """
    path = instruments or dataset_dir(INSTRUMENTS_DATASET) / INSTRUMENTS_FILE
    frame = pd.read_parquet(path)
    parts = cast("pd.Series", frame["symbol"]).str.split(".", expand=True)
    keys = parts[1] + "." + parts[0]
    return pd.Series(cast("pd.Series", frame["name"]).to_numpy(), index=keys)


def match_industry(name: str) -> str | None:
    """按名称匹配行业（先排除宽基/风格/跨境/商品/债，再按具体度匹配）。"""
    if not isinstance(name, str) or not name or EXCLUDE_RE.search(name):
        return None
    for industry, pattern in INDUSTRY_RES:
        if pattern.search(name):
            return industry
    return None


def symbol_name(names: pd.Series, symbol: str) -> str:
    """取标的名（缺失或非字符串时返回空串）。"""
    value = names.get(symbol, "")
    return value if isinstance(value, str) else ""


def build_pool(
    adj: pd.DataFrame,
    amount: pd.DataFrame,
    names: pd.Series,
    classes: dict[str, str],
    *,
    min_bars: int = MIN_BARS,
    min_amount: float = MIN_AMOUNT,
) -> dict[str, str]:
    """行业映射 + 同行业保留最活跃 + 跨行业同质去重，返回 symbol → industry。"""
    liquid = cast("pd.Series", amount.tail(min_bars).mean())
    candidates: list[tuple[str, str, float]] = []
    for symbol in adj.columns:
        if classes.get(symbol) != "A股股票型":
            continue
        industry = match_industry(symbol_name(names, symbol))
        if industry is None or len(adj[symbol].dropna()) < min_bars:
            continue
        turnover = _num(liquid.get(symbol, 0.0))
        if turnover >= min_amount and np.isfinite(turnover):
            candidates.append((symbol, industry, turnover))
    candidates.sort(key=lambda item: -item[2])
    kept: dict[str, str] = {}
    returns = adj.pct_change()
    for symbol, industry, _ in candidates:
        column = cast("pd.Series", returns[symbol])
        if any(industry == kept[other] for other in kept):
            continue
        duplicates = (
            abs(_num(column.corr(cast("pd.Series", returns[other])))) > CORR_DEDUP
            for other in kept
        )
        if any(duplicates):
            continue
        kept[symbol] = industry
    return kept
