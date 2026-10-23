"""ETF 时序趋势跟随（TSMOM）：逐资产绝对趋势判定 + 等权，不做横截面排序。

与 :class:`EtfRotationStrategy` 的区别：
- 不排名、不选数量，对固定 ``basket`` 内每只标的**独立**判定趋势
  （默认双确认：12 个月动量为正 **且** 站上 200 日均线）；
- 达标者按 ``1/N``（N = basket 总数）等权分配，未达标名额留现金——
  总仓位随「在场宽度」自动升缩，这是趋势跟随的崩盘保护来源；
- ``require_momentum`` / ``require_ma`` 两个开关用于对照实验（单条件 vs 双确认）。

面板诊断（``scripts/diagnose_tsmom_panel.py``）显示：宽基/行业/黄金的时序信号有
正区分度，跨境/商品为负、债无效，故默认 basket 只服务 A 股 beta + 黄金类资产。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from app.strategy.native.etf_common import LOT_SIZE, _int, _num
from app.strategy.native.etf_rotation import EtfRotationStrategy


class TimeSeriesTrendStrategy(EtfRotationStrategy):
    """时序趋势跟随：basket 内达标资产等权持有，其余名额留现金。"""

    def __init__(
        self,
        strategy_id: str = "tsmom",
        basket: Sequence[str] = (),
        hold_days: int = 21,
        momentum_window: int = 252,
        ma_window: int = 200,
        require_momentum: bool = True,
        require_ma: bool = True,
        target_exposure: float = 1.0,
        warmup_until: datetime | None = None,
    ) -> None:
        self.basket = list(basket)
        self.require_momentum = require_momentum
        self.require_ma = require_ma
        # 引擎按 watched_symbols 聚焦每日 K 线；只关心 basket，其余标的不喂
        self.watched_symbols = frozenset(self.basket)
        super().__init__(
            strategy_id=strategy_id,
            select_num=max(1, len(self.basket)),
            hold_days=hold_days,
            momentum_window=momentum_window,
            ma_window=ma_window,
            min_amount=0.0,
            min_vol=0.0,
            require_trend=False,
            max_corr=1.0,
            target_exposure=target_exposure,
            warmup_until=warmup_until,
        )

    # ------------------------------------------------------------------ 预热

    def _load_warmup(self, until: datetime) -> None:
        """定向预载 basket 内标的的历史（基类版本会全市场预载）。

        只读取 ``until`` 之前的行（严格 PIT，不引入未来信息），与基类口径一致。
        """
        root = self._market_root()
        if root is None:
            return
        cutoff = pd.Timestamp(until) - pd.Timedelta(days=1)
        keep = self._warmup() + 10
        for symbol in self.basket:
            exchange, _, code = symbol.partition(".")
            if exchange not in ("SH", "SZ") or not code.isdigit():
                continue
            path = root / exchange / f"{code}.parquet"
            if not path.is_file():
                continue
            frame = pd.read_parquet(path, columns=["close", "adj_factor", "amount"])
            frame = frame[frame.index <= cutoff].tail(keep)
            if frame.empty:
                continue
            prices = frame["close"].astype("float64") * frame["adj_factor"].astype(
                "float64"
            )
            self._adj_close[symbol] = [_num(value) for value in prices]
            self._amount[symbol] = [
                _num(value) for value in frame["amount"].astype("float64")
            ]

    # ------------------------------------------------------------------ 信号

    def _trend_on(self, symbol: str) -> bool:
        """单资产趋势判定：动量与均线确认（按开关生效，历史不足视为不达标）。"""
        closes = self._adj_close.get(symbol)
        if not closes or len(closes) < self._warmup():
            return False
        latest = closes[-1]
        if self.require_momentum:
            base = closes[-self.momentum_window - 1]
            if base <= 0 or latest <= base:
                return False
        if self.require_ma:
            window = closes[-self.ma_window :]
            if latest < sum(window) / len(window):
                return False
        return True

    # ------------------------------------------------------------------ 目标持仓

    def _rebalance(self, bars: dict[str, Any], portfolio: Any) -> dict[str, int]:
        """调仓日：basket 内达标者按 1/N 等权，未达标名额留现金。"""
        portfolio_value = _num(getattr(portfolio, "total_value", 0.0))
        if portfolio_value <= 0:
            return self._current_targets(portfolio)
        slot_value = portfolio_value * self.target_exposure / max(1, len(self.basket))
        desired: dict[str, int] = {}
        for symbol in self.basket:
            if not self._trend_on(symbol):
                continue
            bar = bars.get(symbol)
            price = _num(getattr(bar, "close", 0.0)) if bar else 0.0
            if price <= 0:
                continue
            quantity = _int(slot_value / price) // LOT_SIZE * LOT_SIZE
            if quantity > 0:
                desired[symbol] = quantity
        return self._ordered_targets(desired, bars, portfolio, portfolio_value)
