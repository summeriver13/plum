from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import cast

from app.engine.backtest.models import Signal, SignalDirection, SignalType
from app.engine.backtest.portfolio import Portfolio
from app.data.schema import Bar
from app.strategy.base import (
    StatefulMixin,
    WindowedStrategy,
)


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"


@dataclass
class TrendFollowingStrategy(WindowedStrategy, StatefulMixin):
    ema_fast: int = 14
    ema_slow: int = 25
    ma_trend: int = 200
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14

    adx_threshold: float = 30.0
    rsi_oversold: float = 30.0
    atr_multiplier: float = 3.0
    hard_stop_pct: float = 0.05

    # 市场状态检测
    trend_buffer: float = 0.02  # 偏离MA超过此比例才切换状态

    # BULL参数
    bull_max_hold_days: int = 60  # 仅作异常保护，不常规触发

    # BEAR参数
    bear_max_hold_days: int = 20  # 均值回归策略需控制时间风险
    adx_trend_threshold_bear: float = 25.0  # ADX低于此值才允许逆势入场

    max_positions: int = 3
    can_short: bool = False

    _regime: MarketRegime = MarketRegime.BULL

    def __post_init__(self) -> None:
        WindowedStrategy.__init__(
            self, window=max(self.ma_trend, self.bb_period, self.ema_slow)
        )

    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> list[Signal]:
        self.update_history(bars)

        if not self.is_ready:
            return []

        signals: list[Signal] = []
        current_positions = {s: p.quantity for s, p in portfolio.positions.items()}

        # 时间止损：状态感知（BULL/BEAR分隔max_hold_days）
        last_signal = cast(dict[str, int], self.get_state_value("_last_signal", {}))
        atr_trailing = cast(
            dict[str, Decimal], self.get_state_value("_atr_trailing", {})
        )
        entry_regimes = cast(
            dict[str, MarketRegime], self.get_state_value("_entry_regime", {})
        )

        time_stop_signals: list[Signal] = []
        for symbol, qty in current_positions.items():
            if qty > 0:
                entry_regime = entry_regimes.get(symbol, MarketRegime.BULL)
                max_days = (
                    self.bull_max_hold_days
                    if entry_regime == MarketRegime.BULL
                    else self.bear_max_hold_days
                )
                hold_days = portfolio.get_hold_days(symbol, ts)
                if hold_days is not None and hold_days >= max_days:
                    time_stop_signals.append(
                        Signal(
                            symbol=symbol,
                            direction=SignalDirection.FLAT,
                            signal_type=SignalType.EXIT,
                            reason="MAX_HOLD_DAYS",
                            source=f"{self.__class__.__name__}.TIME_STOP",
                            timestamp=ts,
                        )
                    )

        # 应用时间止损信号
        time_stop_closed: set[str] = set()
        for sig in time_stop_signals:
            time_stop_closed.add(sig.symbol)
            last_signal[sig.symbol] = -1
            atr_trailing.pop(sig.symbol, None)
            entry_regimes.pop(sig.symbol, None)
            current_positions.pop(sig.symbol, None)
            self.update_state("_last_signal", last_signal)
            self.update_state("_atr_trailing", atr_trailing)
            self.update_state("_entry_regime", entry_regimes)
        signals.extend(time_stop_signals)

        for symbol in bars:
            ind = self._calculate_indicators(symbol)
            if ind is None:
                continue

            regime = self._detect_regime(ind)
            self._regime = regime

            last_signal = cast(dict[str, int], self.get_state_value("_last_signal", {}))
            atr_trailing = cast(
                dict[str, Decimal], self.get_state_value("_atr_trailing", {})
            )
            entry_regimes = cast(
                dict[str, MarketRegime], self.get_state_value("_entry_regime", {})
            )
            prev_below = cast(
                dict[str, bool], self.get_state_value("_prev_below_lower", {})
            )
            prev_adx = cast(dict[str, float], self.get_state_value("_prev_adx", {}))

            has_position = current_positions.get(symbol, 0) > 0

            if regime == MarketRegime.BULL:
                signal = self._bull_signals(
                    ts,
                    symbol,
                    ind,
                    has_position,
                    atr_trailing,
                    last_signal,
                    time_stop_closed,
                )
            else:
                signal = self._bear_signals(
                    ts,
                    symbol,
                    ind,
                    has_position,
                    portfolio,
                    last_signal,
                    prev_below,
                )

            if signal is not None:
                signals.append(signal)

                if signal.direction == SignalDirection.FLAT and has_position:
                    last_signal[symbol] = -1
                    atr_trailing.pop(symbol, None)
                    entry_regimes.pop(symbol, None)
                    self.update_state("_last_signal", last_signal)
                    self.update_state("_atr_trailing", atr_trailing)
                    self.update_state("_entry_regime", entry_regimes)
                elif signal.direction == SignalDirection.LONG:
                    atr_trailing[symbol] = ind["price"]
                    entry_regimes[symbol] = regime
                    last_signal[symbol] = 1
                    self.update_state("_atr_trailing", atr_trailing)
                    self.update_state("_entry_regime", entry_regimes)
                    self.update_state("_last_signal", last_signal)

            # 更新prev_below_lower状态用于BEAR状态恢复检测
            below_lower = ind["price"] < ind["bb_lower"]
            prev_below[symbol] = below_lower
            self.update_state("_prev_below_lower", prev_below)
            prev_adx[symbol] = ind["adx"]
            self.update_state("_prev_adx", prev_adx)

        return signals

    def _calculate_indicators(self, symbol: str) -> dict | None:
        """计算技术指标（简化版，使用内置MA/EMA）。"""
        history = self.get_history(symbol)
        if len(history) < self.ma_trend:
            return None

        price = history[-1].close
        ma = self.ma(symbol, self.ma_trend)
        ema_fast = self.ema(symbol, self.ema_fast)
        ema_slow = self.ema(symbol, self.ema_slow)

        # 简化ATR计算
        atr = self._calculate_atr(symbol)

        # 简化布林带计算
        bb_middle = ma
        bb_std_dev = self._calculate_std_dev(symbol, self.bb_period)
        bb_upper = bb_middle + Decimal(str(self.bb_std)) * bb_std_dev
        bb_lower = bb_middle - Decimal(str(self.bb_std)) * bb_std_dev

        # 简化RSI计算
        rsi = self._calculate_rsi(symbol, self.rsi_period)

        # 简化ADX计算
        adx = self._calculate_adx(symbol, self.adx_period)

        return {
            "price": price,
            "ma": ma,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "atr": atr,
            "bb_middle": bb_middle,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "rsi": rsi,
            "adx": adx,
        }

    def _calculate_atr(self, symbol: str, period: int | None = None) -> Decimal:
        """计算平均真实波动范围。"""
        if period is None:
            period = self.atr_period
        history = self.get_history(symbol)
        if len(history) < period + 1:
            return Decimal("0")

        true_ranges = []
        for i in range(len(history) - period, len(history)):
            bar = history[i]
            prev_bar = history[i - 1]
            high_low = bar.high - bar.low
            high_close = abs(bar.high - prev_bar.close)
            low_close = abs(bar.low - prev_bar.close)
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)

        return sum(true_ranges) / Decimal(period)

    def _calculate_std_dev(self, symbol: str, period: int) -> Decimal:
        """计算标准差。"""
        history = self.get_history(symbol)
        if len(history) < period:
            return Decimal("0")

        prices = [b.close for b in history[-period:]]
        mean = sum(prices) / Decimal(period)
        variance = sum((p - mean) ** 2 for p in prices) / Decimal(period)
        return variance.sqrt()

    def _calculate_rsi(self, symbol: str, period: int) -> float:
        """计算RSI指标。"""
        history = self.get_history(symbol)
        if len(history) < period + 1:
            return 50.0

        gains = []
        losses = []
        for i in range(len(history) - period, len(history)):
            change = float(history[i].close - history[i - 1].close)
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _calculate_adx(self, symbol: str, period: int) -> float:
        """计算ADX指标（简化版）。"""
        history = self.get_history(symbol)
        if len(history) < period + 1:
            return 25.0

        # 简化：使用价格变动幅度作为趋势强度代理
        changes = []
        for i in range(len(history) - period, len(history)):
            change = abs(float(history[i].close - history[i - 1].close))
            changes.append(change)

        avg_change = sum(changes) / len(changes)
        # 归一化到0-50范围
        return min(50.0, avg_change * 100)

    def _detect_regime(self, ind: dict) -> MarketRegime:
        """检测市场状态，带缓冲区避免边界抖动。"""
        ma = ind["ma"]
        price = ind["price"]
        prev_regime = cast(
            MarketRegime, self.get_state_value("_prev_regime", MarketRegime.BULL)
        )

        bull_threshold = ma * (Decimal("1") + Decimal(str(self.trend_buffer)))
        bear_threshold = ma * (Decimal("1") - Decimal(str(self.trend_buffer)))

        if price > bull_threshold:
            detected = MarketRegime.BULL
        elif price < bear_threshold:
            detected = MarketRegime.BEAR
        else:
            detected = prev_regime  # 保持当前状态

        # 持久化用于下一bar
        self.update_state("_prev_regime", detected)
        return detected

    def _bull_signals(
        self,
        ts: datetime,
        symbol: str,
        ind: dict,
        has_position: bool,
        atr_trailing: dict[str, Decimal],
        last_signal: dict[str, int],
        time_stop_closed: set[str],
    ) -> Signal | None:
        ema_fast = ind["ema_fast"]
        ema_slow = ind["ema_slow"]
        adx = ind["adx"]
        price = ind["price"]
        atr = ind["atr"]

        if not has_position:
            # 跳过TIME_STOP已平仓的股票
            if symbol in time_stop_closed:
                return None
            # EMA交叉足够用于趋势跟踪入场
            if ema_fast > ema_slow:
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    signal_type=SignalType.ENTRY,
                    strength=min(adx / 50.0, 1.0),
                    reason="EMA_CROSS",
                    source=f"{self.__class__.__name__}.BULL",
                    timestamp=ts,
                )
            return None

        # ATR自适应 trailing stop
        peak = atr_trailing.get(symbol, price)
        if price > peak:
            peak = price
            atr_trailing[symbol] = peak
            self.update_state("_atr_trailing", atr_trailing)

        stop_level = peak - Decimal(str(self.atr_multiplier)) * atr
        if price < stop_level:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.FLAT,
                signal_type=SignalType.STOP_LOSS,
                reason="ATR_TRAILING_STOP",
                source=f"{self.__class__.__name__}.BULL",
                timestamp=ts,
            )

        return None

    def _bear_signals(
        self,
        ts: datetime,
        symbol: str,
        ind: dict,
        has_position: bool,
        portfolio: Portfolio,
        last_signal: dict[str, int],
        prev_below_lower: dict[str, bool],
    ) -> Signal | None:
        price = ind["price"]
        bb_middle = ind["bb_middle"]
        bb_lower = ind["bb_lower"]
        rsi = ind["rsi"]
        adx = ind["adx"]

        if not has_position:
            if not self.can_short and last_signal.get(symbol, 0) > 0:
                return None

            below_lower = price < bb_lower
            oversold = rsi < self.rsi_oversold
            # ADX过滤：禁止强下跌趋势中逆势入场
            strong_trend = adx >= self.adx_trend_threshold_bear
            # 恢复检测：从BB下轨下方穿越到上方
            recovered = prev_below_lower.get(symbol, False) and not below_lower

            if (below_lower or recovered) and oversold and not strong_trend:
                strength = (self.rsi_oversold - rsi) / self.rsi_oversold
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    signal_type=SignalType.ENTRY,
                    strength=max(0.5, strength),
                    reason="BB_TOUCH + RSI_OVERSOLD + RECOVERY + ADX_FILTER",
                    source=f"{self.__class__.__name__}.BEAR",
                    timestamp=ts,
                )
            return None

        current_hold_days = portfolio.get_hold_days(symbol, ts)
        pos = portfolio.get_position(symbol)
        entry_price = pos.avg_cost if pos else price
        stop_loss = entry_price * (Decimal("1") - Decimal(str(self.hard_stop_pct)))

        if price < stop_loss:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.FLAT,
                signal_type=SignalType.STOP_LOSS,
                reason="HARD_STOP",
                source=f"{self.__class__.__name__}.BEAR",
                timestamp=ts,
            )

        if current_hold_days >= self.bear_max_hold_days:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.FLAT,
                signal_type=SignalType.EXIT,
                reason="MAX_HOLD_DAYS",
                source=f"{self.__class__.__name__}.BEAR",
                timestamp=ts,
            )

        if price >= bb_middle:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.FLAT,
                signal_type=SignalType.TAKE_PROFIT,
                reason="MEAN_REVERSION_TARGET",
                source=f"{self.__class__.__name__}.BEAR",
                timestamp=ts,
            )

        return None
