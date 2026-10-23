"""ETF 候选池筛选：白名单、相关性分类、流动性/波动准入、趋势门槛、同质去重。

为什么用相关性分类而不是代码段清单
    市场副本只有行情、没有分类元数据；按代码段人工排除既难维护又会误伤。
    TSMOM 面板诊断（2026-09-19）已证明「时序信号只在 A 股宽基/行业与黄金上有正
    区分度，跨境/商品反向、债无效」，于是把「是不是 A 股股票型」定义成可观测事实：
    与 A 股宽基参照 ETF 的 60 日收益相关性 ≥ ``ref_min_corr``（默认关闭 =
    保持 v1~v3 批次的全池口径可复现）。

PIT 说明
    参照序列在 ``__init__`` 一次性读入整份文件，但只按当前 K 线时间戳取值
    （``_record_refs``）：任意决策时刻只用到 ≤ 当日的收盘价，不引入未来信息。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from app.strategy.native.etf_common import (
    AMOUNT_WINDOW,
    CORR_WINDOW,
    VOL_WINDOW,
    _num,
    _returns,
    _stdev,
)
from app.strategy.native.etf_warmup import EtfWarmupMixin

DEFAULT_REF_SYMBOLS: tuple[str, ...] = ("SH.510300", "SH.510500", "SH.512100")
"""A 股宽基参照 ETF（沪深 300 / 中证 500 / 中证 1000），历史均可回溯到 2012 年前。"""

DEFAULT_REF_CORR_WINDOW = 60
DEFAULT_REF_MIN_CORR = 0.6


class EtfSelectionMixin(EtfWarmupMixin):
    """候选池准入判断（无独立状态，依赖策略侧注入的参数与历史缓存）。"""

    universe: Sequence[str]
    ref_symbols: Sequence[str]
    ref_corr_window: int
    ref_min_corr: float
    min_amount: float
    min_vol: float
    ma_window: int
    trend_buffer: float
    max_corr: float
    _adj_close: dict[str, list[float]]
    _amount: dict[str, list[float]]
    _ref_series: dict[str, pd.Series]
    _ref_close: dict[str, list[float]]

    # ------------------------------------------------------------------ 相关性分类

    def _load_refs(self) -> None:
        """读入参照 ETF 的后复权收盘序列（按时间戳取值，避免未来函数）。"""
        self._ref_series = {}
        self._ref_close = {symbol: [] for symbol in self.ref_symbols}
        root = self._market_root()
        if root is None or not self.ref_symbols:
            return
        for symbol in self.ref_symbols:
            exchange, _, code = symbol.partition(".")
            if exchange not in ("SH", "SZ") or not code.isdigit():
                continue
            path = root / exchange / f"{code}.parquet"
            if not path.is_file():
                continue
            frame = pd.read_parquet(path, columns=["close", "adj_factor"])
            prices = frame["close"].astype("float64") * frame["adj_factor"].astype(
                "float64"
            )
            self._ref_series[symbol] = prices

    def _record_refs(self, ts: datetime) -> None:
        """推进当日参照收盘（缺该时间戳则跳过，保持与 K 线序列对齐）。"""
        for symbol, series in self._ref_series.items():
            value = _num(series.get(pd.Timestamp(ts)), math.nan)
            if not math.isnan(value):
                self._ref_close[symbol].append(value)

    def _passes_universe(self, symbol: str) -> bool:
        """白名单（若设置）与 A 股股票型相关性分类（未开启则只查白名单）。"""
        if self.universe and symbol not in self.universe:
            return False
        if self.ref_min_corr <= 0.0 or not self._ref_series:
            return True
        returns = _returns(self._adj_close[symbol])
        return self._best_ref_corr(returns) >= self.ref_min_corr

    def _best_ref_corr(self, returns: list[float]) -> float:
        """与各参照序列的 60 日收益相关性的最大值（数据不足视为不达标 = 负值）。"""
        window = self.ref_corr_window
        best = -1.0
        for closes in self._ref_close.values():
            ref = _returns(closes)
            size = min(len(returns), len(ref))
            if size < window:
                continue
            recent, ref_recent = returns[-window:], ref[-size:][-window:]
            if _stdev(recent) == 0 or _stdev(ref_recent) == 0:
                continue
            best = max(best, statistics.correlation(recent, ref_recent))
        return best

    # ------------------------------------------------------------------ 其他准入

    def _is_liquid(self, symbol: str) -> bool:
        """流动性准入：20 日均成交额达标，且（可选）波动率不低于门槛。"""
        amounts = self._amount.get(symbol, [])
        if len(amounts) < AMOUNT_WINDOW:
            return False
        if sum(amounts[-AMOUNT_WINDOW:]) / AMOUNT_WINDOW < self.min_amount:
            return False
        if self.min_vol <= 0:
            return True
        closes = self._adj_close[symbol]
        vol = _stdev(_returns(closes[-(VOL_WINDOW + 1) :]))
        return vol >= self.min_vol

    def _passes_trend(self, symbol: str) -> bool:
        """绝对趋势过滤：后复权价站在 MA 之上（可加缓冲）。"""
        closes = self._adj_close[symbol]
        window = closes[-self.ma_window :]
        average = sum(window) / len(window)
        return closes[-1] >= average * (1.0 + self.trend_buffer)

    def _is_homogeneous(self, symbol: str, chosen: list[str]) -> bool:
        """与已选标的高度相关（同指数不同 ETF）则视为同质。"""
        if self.max_corr >= 1.0:
            return False
        recent = _returns(self._adj_close[symbol][-(CORR_WINDOW + 1) :])
        for other in chosen:
            peer = _returns(self._adj_close[other][-(CORR_WINDOW + 1) :])
            size = min(len(recent), len(peer))
            if size < 20 or _stdev(recent[-size:]) == 0 or _stdev(peer[-size:]) == 0:
                continue
            if statistics.correlation(recent[-size:], peer[-size:]) > self.max_corr:
                return True
        return False
