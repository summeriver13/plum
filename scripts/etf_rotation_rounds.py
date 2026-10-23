"""ETF 双动量轮动的批次轮次定义。

批次：
- v1 策略层五轮（频率/过滤/分散）
- v2 方向与窗口三轮（反转 60 日 / 动量 20 日 / 动量 250 日）
- v3 反转短窗与长窗分散三轮
- v4 标的池与门槛五轮（单只基线 / 相关性分类受池 × MA200 × top3/top5 / 无门槛对照）

独立成模块的原因：轮次定义随批次持续增长，执行器（backtest_etf_rotation.py）保持精简。
注意：本模块导入 app.*，调用方需在导入前设定 PLUM_DATA_MARKET_PATH（与执行器同一约定）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.strategy.native.etf_rotation import DEFAULT_MAX_CORR  # noqa: E402
from app.strategy.native.etf_selection import DEFAULT_REF_MIN_CORR  # noqa: E402


@dataclass(frozen=True)
class EtfRound:
    """一轮 ETF 轮动实验的配置。"""

    round_id: int
    title: str
    hypothesis: str
    select_num: int = 3
    hold_days: int = 10
    momentum_window: int = 60
    ma_window: int = 120
    require_trend: bool = True  # False = 纯相对动量对照组
    reversal: bool = False  # True = 买动量最弱（反向验证）
    min_amount: float = 1.0e7  # 20 日均成交额门槛（元）
    max_corr: float = DEFAULT_MAX_CORR  # 同质去重相关性阈值
    universe: tuple[str, ...] = ()  # 白名单（非空时只在此范围内选）
    ref_min_corr: float = 0.0  # >0 时启用「与 A 股宽基参照的相关性」分类（0 = 全池）


ROUNDS: list[EtfRound] = [
    EtfRound(
        round_id=1,
        title="纯相对动量（无绝对过滤）",
        hypothesis="只看横截面动量强弱、不做趋势过滤，满仓风险资产；作为双动量的对照组",
        require_trend=False,
    ),
    EtfRound(
        round_id=2,
        title="双动量（相对 + MA120 绝对过滤）",
        hypothesis="加入绝对趋势过滤后，走坏的风格名额留现金，预期显著压低回撤且保留大部分收益",
    ),
    EtfRound(
        round_id=3,
        title="5 日调仓",
        hypothesis="短周期更快响应风格切换，代价是摩擦与同质去重后的频繁换手",
        hold_days=5,
    ),
    EtfRound(
        round_id=4,
        title="20 日调仓",
        hypothesis="长周期降低摩擦，代价是退出更慢、回撤可能变大",
        hold_days=20,
    ),
    EtfRound(
        round_id=5,
        title="持仓 5 只分散",
        hypothesis="分散到 5 只降低单一风格风险，代价是摊薄最强动量的收益",
        select_num=5,
    ),
]

# 第二批（v2）：诊断结论是「60 日动量选股的前向 10 日收益中位为负」，
# 因此固定 10 日调仓/3 只/MA120 过滤，只扫描方向与窗口两个维度。
V2_ROUNDS: list[EtfRound] = [
    EtfRound(
        round_id=11,
        title="反转 60 日（方向翻转）",
        hypothesis=(
            "60 日动量选前 3 名的前向 10 日中位收益 -0.24%（全池 +0.16%），"
            "若短周期反转主导，则买动量最弱 3 只应当转正"
        ),
        reversal=True,
    ),
    EtfRound(
        round_id=12,
        title="动量 20 日（短窗）",
        hypothesis="若劣势来自 60 日窗口跟不上脉冲衰减，缩短到 20 日应当改善",
        momentum_window=20,
    ),
    EtfRound(
        round_id=13,
        title="动量 250 日（长窗，经典配方）",
        hypothesis=(
            "经典 ETF 动量用 12 个月窗口：若 60 日的暴跌是窗口不当而非方向错误，"
            "长窗应恢复正收益"
        ),
        momentum_window=250,
    ),
]


# 第三批（v3）：v2 的结论是「方向与窗口都不是唯一病灶」——
# 反转 60 日与短窗动量均未转正，只有 250 日长窗为正（Sharpe 0.34）。
# 本批同时改「信号」（把反转窗压到 20/5 日，验证均值回复是否只在极短周期存在）
# 与「结构」（长窗动量配 5 只分散，降低单一风格暴露）。
V3_ROUNDS: list[EtfRound] = [
    EtfRound(
        round_id=21,
        title="反转 20 日",
        hypothesis="若反转收益集中在 1 个月以内，20 日窗应比 60 日更接近拐点",
        reversal=True,
        momentum_window=20,
    ),
    EtfRound(
        round_id=22,
        title="反转 5 日",
        hypothesis="检验一周级别的均值回复；若仍不能转正，则反转方向整体证伪",
        reversal=True,
        momentum_window=5,
    ),
    EtfRound(
        round_id=23,
        title="动量 250 日 + 5 只分散",
        hypothesis="对唯一为正的 250 日动量降单风格暴露，检验 Sharpe 能否提高",
        momentum_window=250,
        select_num=5,
    ),
]

# 第四批（v4）：前三批的结论是「250 日窗是唯一正窗口，但 IR 只有 0.19」。
# 本批换「池子口径」这个从未测过的维度：v1~v3 一直是全池（含跨境/商品/货币），
# 而 TSMOM 面板诊断显示时序 edge 只存在于 A 股宽基/行业与黄金；
# 同时补上「不轮动」的单只基线——轮动若赢不了它，就没有存在理由。
V4_ROUNDS: list[EtfRound] = [
    EtfRound(
        round_id=31,
        title="单只宽基 × MA200 门槛（不轮动基线）",
        hypothesis="512100 中证 1000 ETF：站上 MA200 满仓、否则空仓；轮动的比较基准",
        select_num=1,
        universe=("SH.512100",),
        momentum_window=250,  # 单标的时不影响排名，只用于对齐预热长度
        ma_window=200,
    ),
    EtfRound(
        round_id=32,
        title="受池 × 250 日 × MA200 × top3",
        hypothesis=(
            "把与 A 股宽基相关性 < 0.6 的跨境/商品/债剔出候选池，"
            "再配 250 日排名与 MA200 门槛，检验池子修好后能否提高 IR"
        ),
        momentum_window=250,
        ma_window=200,
        select_num=3,
        ref_min_corr=DEFAULT_REF_MIN_CORR,
    ),
    EtfRound(
        round_id=33,
        title="受池 × 250 日 × MA200 × top5",
        hypothesis="在 R32 基础上分散到 5 只，检验分散是否仍能提高 Sharpe（v3 R23 的经验）",
        momentum_window=250,
        ma_window=200,
        select_num=5,
        ref_min_corr=DEFAULT_REF_MIN_CORR,
    ),
    EtfRound(
        round_id=34,
        title="受池 × 250 日 × 无门槛 × top5",
        hypothesis="池子修好后是否不再需要绝对趋势门槛（隔离门槛的边际贡献）",
        momentum_window=250,
        ma_window=200,
        select_num=5,
        require_trend=False,
        ref_min_corr=DEFAULT_REF_MIN_CORR,
    ),
    EtfRound(
        round_id=35,
        title="全池 × 250 日 × MA200 × top5（门槛参数对照）",
        hypothesis="只把门槛从 MA120 换成 MA200（池子仍全池），与 R23 对比隔离两个变量",
        momentum_window=250,
        ma_window=200,
        select_num=5,
    ),
    EtfRound(
        round_id=36,
        title="全池 × 250 日 × 无门槛 × top5（2×2 补格）",
        hypothesis="与 R35 成对：全池下门槛的边际贡献（池子×门槛 2×2 的最后格）",
        momentum_window=250,
        ma_window=200,
        select_num=5,
        require_trend=False,
    ),
    EtfRound(
        round_id=37,
        title="单只宽基 × MA200 × 21 日调仓（基线稳健性）",
        hypothesis="R31 的 10 日调仓可能被震荡打脸；改为月度检查看单只基线会不会变高",
        select_num=1,
        universe=("SH.512100",),
        hold_days=21,
        momentum_window=250,
        ma_window=200,
    ),
]

BATCH_ROUNDS: dict[str, list[EtfRound]] = {
    "v1": ROUNDS,
    "v2": V2_ROUNDS,
    "v3": V3_ROUNDS,
    "v4": V4_ROUNDS,
}
