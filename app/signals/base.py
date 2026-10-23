"""信号适配器 — 将择时信号实现包装为 Plum Signal 列表。"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd

# NaN 常量：避免每处裸写 float("nan") 构造调用
NOT_A_NUMBER: float = math.nan


def _get_signal_types():
    """延迟导入 Signal 相关类型，避免触发 data 包的全量加载。"""
    from app.engine.backtest.models import Signal, SignalDirection, SignalType

    return Signal, SignalDirection, SignalType


class TimingSignalAdapter:
    """大盘/个股择时信号适配器基类。"""

    def __init__(self, name: str = "timing") -> None:
        self.name = name

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        raise NotImplementedError

    def to_plum_signals(
        self,
        timing_result: pd.Series | pd.DataFrame,
        ts: datetime,
        bars: dict[str, Any],
        selected_symbols: list[str] | None = None,
    ) -> list[Any]:
        Signal, SignalDirection, SignalType = _get_signal_types()

        if isinstance(timing_result, pd.DataFrame):
            if "择时信号" in timing_result.columns:
                signal_value = timing_result["择时信号"].iloc[-1]
            else:
                signal_value = timing_result.iloc[-1, 0]
        else:
            signal_value = timing_result.iloc[-1]

        is_long = bool(signal_value) and signal_value > 0
        targets = selected_symbols or list(bars.keys())

        signals = []
        if is_long:
            for symbol in targets:
                if symbol in bars:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            direction=SignalDirection.LONG,
                            signal_type=SignalType.ENTRY,
                            strength=1.0,
                            reason="择时信号=1",
                            source=self.name,
                            timestamp=ts,
                        )
                    )
        else:
            for symbol in targets:
                signals.append(
                    Signal(
                        symbol=symbol,
                        direction=SignalDirection.FLAT,
                        signal_type=SignalType.EXIT,
                        strength=0.0,
                        reason="择时信号=0",
                        source=self.name,
                        timestamp=ts,
                    )
                )
        return signals


class MarketTimingV1p5(TimingSignalAdapter):
    """大盘择时 v1.5 — 早盘+隔夜下跌比例择时。

    大盘择时信号 v1p5。
    """

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__(name="market_timing_v1p5")
        self.threshold = threshold

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """计算大盘择时信号。"""
        open_col = kwargs.get("open_col", "开盘至今涨幅")
        ovn_col = kwargs.get("ovn_col", "隔夜涨跌幅")

        if open_col in df.columns and ovn_col in df.columns:
            open_decl = df.groupby("交易日期")[open_col].apply(lambda x: (x < 0).mean())
            ovn_decl = df.groupby("交易日期")[ovn_col].apply(lambda x: (x < 0).mean())
            decl_ratio = open_decl + ovn_decl
            return (decl_ratio <= self.threshold).astype(int)
        return pd.Series(dtype=int)


class StockTimingMA(TimingSignalAdapter):
    """个股择时_均线 — 价格偏离均线择时。

    个股均线择时信号。
    """

    def __init__(self, ma_period: int = 12, threshold: float = 0.0) -> None:
        super().__init__(name="stock_timing_ma")
        self.ma_period = ma_period
        self.threshold = threshold

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """计算个股均线偏离度择时信号。

        Args:
            df: 包含 '收盘价_复权' 的 DataFrame

        Returns:
            择时信号 Series (1=做多, 0=空仓)
        """
        if "收盘价_复权" not in df.columns:
            return pd.Series(dtype=int)

        ma_col = f"MA_{self.ma_period}"
        df[ma_col] = df["收盘价_复权"].rolling(self.ma_period).mean()
        bias = df["收盘价_复权"] / df[ma_col] - 1
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[bias > self.threshold] = 1
        return signal.fillna(0).astype(int)


class StockTimingMACD(TimingSignalAdapter):
    """个股择时_MACD — MACD 金叉死叉择时。"""

    def __init__(self) -> None:
        super().__init__(name="stock_timing_macd")

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        if "收盘价_复权" not in df.columns:
            return pd.Series(dtype=int)

        ema12 = df["收盘价_复权"].ewm(span=12, adjust=False).mean()
        ema26 = df["收盘价_复权"].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[dif > dea] = 1
        return signal.fillna(0).astype(int)


# ═══════════════════════════════════════════════════════════════
# 新增 9 个择时信号 (2026-05)
# ═══════════════════════════════════════════════════════════════


class StockTimingKDJ(TimingSignalAdapter):
    """个股择时_KDJ — KDJ 金叉/死叉/死叉发散信号。"""

    def __init__(self, mode: str = "金叉") -> None:
        super().__init__(name=f"stock_timing_kdj_{mode}")
        self.mode = mode

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        if "收盘价_复权" not in df.columns:
            return pd.Series(dtype=int)

        high = df["最高价_复权"] if "最高价_复权" in df.columns else df["收盘价_复权"]
        low = df["最低价_复权"] if "最低价_复权" in df.columns else df["收盘价_复权"]
        close = df["收盘价_复权"]

        low_9 = low.rolling(9, min_periods=9).min()
        high_9 = high.rolling(9, min_periods=9).max()
        rsv = (close - low_9) / (high_9 - low_9) * 100
        rsv = rsv.fillna(50)

        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        signal = pd.Series(0, index=df.index, dtype=int)

        if self.mode == "金叉":
            cond = (k > d) & (k.shift(1) <= d.shift(1))
            signal[cond] = 1
        elif self.mode == "死叉":
            cond = (k < d) & (k.shift(1) >= d.shift(1))
            signal[cond] = 1
        elif self.mode == "死叉发散":
            cond = (d > k) & (k > j)
            cond &= (d.shift(1) > d) & (k.shift(1) > k) & (j.shift(1) > j)
            signal[cond] = 1

        return signal.fillna(0).astype(int)


class StockTimingBollinger(TimingSignalAdapter):
    """个股择时_布林带 — 价格突破布林带上/下轨信号。"""

    def __init__(self, n: int = 8, std_mult: float = 0.5) -> None:
        super().__init__(name="stock_timing_bollinger")
        self.n = n
        self.std_mult = std_mult

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        if "收盘价_复权" not in df.columns:
            return pd.Series(dtype=int)

        close = df["收盘价_复权"]
        ma = close.rolling(self.n, min_periods=1).mean()
        std = close.rolling(self.n, min_periods=1).std()

        upper = ma + self.std_mult * std
        lower = ma - self.std_mult * std

        signal = pd.Series(NOT_A_NUMBER, index=df.index)
        signal.loc[close > upper] = 1.0
        signal.loc[close < lower] = 0.0

        return signal.ffill().fillna(0).astype(int)


class StockTimingVolume(TimingSignalAdapter):
    """个股择时_成交量 — 成交量均线金叉/死叉信号。"""

    def __init__(self, mode: str = "金叉", short_n: int = 5, long_n: int = 10) -> None:
        super().__init__(name=f"stock_timing_vol_{mode}")
        self.mode = mode
        self.short_n = short_n
        self.long_n = long_n

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        vol = None
        for c in ["成交量", "成交额"]:
            if c in df.columns:
                vol = df[c]
                break
        if vol is None:
            return pd.Series(0, index=df.index, dtype=int)

        ma_short = vol.rolling(self.short_n, min_periods=1).mean()
        ma_long = vol.rolling(self.long_n, min_periods=1).mean()

        signal = pd.Series(0, index=df.index, dtype=int)

        if self.mode == "金叉":
            cond = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))
            signal[cond] = 1
        elif self.mode == "死叉":
            cond = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
            signal[cond] = 1
        elif self.mode == "缩量":
            signal[ma_short < ma_long] = 1

        return signal.fillna(0).astype(int)


class StockTimingTurtle(TimingSignalAdapter):
    """个股择时_海龟 — 突破N日最高开仓，跌破N日最低平仓。"""

    def __init__(self, n: int = 4) -> None:
        super().__init__(name="stock_timing_turtle")
        self.n = n

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        if "收盘价_复权" not in df.columns:
            return pd.Series(dtype=int)

        close = df["收盘价_复权"]
        high_n = close.rolling(self.n, min_periods=1).max().shift(1)
        low_n = close.rolling(self.n, min_periods=1).min().shift(1)

        signal = pd.Series(NOT_A_NUMBER, index=df.index)
        signal[close > high_n] = 1.0
        signal[close < low_n] = 0.0

        return signal.ffill().fillna(0).astype(int)


class StockTimingMAEnhanced(TimingSignalAdapter):
    """个股择时_均线增强 — 多模式均线信号。"""

    def __init__(
        self, mode: str = "金叉", short_n: int = 5, long_n: int = 10, pct: float = 2.5
    ) -> None:
        super().__init__(name=f"stock_timing_ma_enh_{mode}")
        self.mode = mode
        self.short_n = short_n
        self.long_n = long_n
        self.pct = pct

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        if "收盘价_复权" not in df.columns:
            return pd.Series(dtype=int)

        close = df["收盘价_复权"]
        signal = pd.Series(0, index=df.index, dtype=int)

        if self.mode == "接近均线":
            ma = close.rolling(self.short_n, min_periods=1).mean()
            lower = ma * (1 - self.pct / 100)
            upper = ma * (1 + self.pct / 100)
            signal[(close >= lower) & (close <= upper)] = 1

        elif self.mode == "跌破均线":
            ma = close.rolling(self.short_n, min_periods=1).mean()
            target = ma * (1 - self.pct / 100)
            signal[close <= target] = 1

        elif self.mode == "金叉":
            ma_s = close.rolling(self.short_n, min_periods=1).mean()
            ma_l = close.rolling(self.long_n, min_periods=1).mean()
            cond = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
            signal[cond] = 1

        elif self.mode == "死叉":
            ma_s = close.rolling(self.short_n, min_periods=1).mean()
            ma_l = close.rolling(self.long_n, min_periods=1).mean()
            cond = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))
            signal[cond] = 1

        elif self.mode == "短期小于长期":
            ma_s = close.rolling(self.short_n, min_periods=1).mean()
            ma_l = close.rolling(self.long_n, min_periods=1).mean()
            signal[ma_s < ma_l] = 1

        elif self.mode == "涨幅破均线":
            prev = close.shift(1).replace(0, NOT_A_NUMBER)
            pct_chg = (close / prev - 1) * 100
            ma_5 = close.rolling(5, min_periods=1).mean()
            signal[(pct_chg > 5) & (close < ma_5)] = 1

        return signal.fillna(0).astype(int)


class StockTimingBollingerEnhanced(TimingSignalAdapter):
    """个股择时_布林带增强 — 触及布林上轨/直接买入信号。"""

    def __init__(self, mode: str = "触及上轨", pct: float = 0.975) -> None:
        super().__init__(name=f"stock_timing_bb_enh_{mode}")
        self.mode = mode
        self.pct = pct

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        if "收盘价_复权" not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)

        close = df["收盘价_复权"]
        n, k = 20, 2
        mid = close.rolling(n, min_periods=1).mean()
        std = close.rolling(n, min_periods=1).std(ddof=0)
        upper = mid + k * std

        signal = pd.Series(0, index=df.index, dtype=int)

        if self.mode == "触及上轨":
            signal[close >= upper * self.pct] = 1
        elif self.mode == "直接买入":
            signal[:] = 1

        return signal.fillna(0).astype(int)


class StockTimingGeneric(TimingSignalAdapter):
    """个股择时_通用 — 11种通用形态与价格信号。"""

    def __init__(
        self, mode: str = "达到目标价", target: float = 100.0, pct: float = 2.0
    ) -> None:
        super().__init__(name=f"stock_timing_gen_{mode}")
        self.mode = mode
        self.target = target
        self.pct = pct

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        close = df.get("收盘价_复权", df.get("收盘价"))
        high = df.get("最高价_复权", df.get("最高价"))
        open_p = df.get("开盘价_复权", df.get("开盘价"))

        if close is None:
            return pd.Series(0, index=df.index, dtype=int)

        signal = pd.Series(0, index=df.index, dtype=int)
        prev_close = close.shift(1)

        if self.mode == "达到目标价":
            signal[close >= self.target] = 1
        elif self.mode == "低于目标价":
            signal[close <= self.target] = 1
        elif self.mode == "冲高回落" and high is not None:
            pct_chg = (high / prev_close.replace(0, NOT_A_NUMBER) - 1) * 100
            signal[(pct_chg > 5) & (close < high * (1 - self.pct / 100))] = 1
        elif self.mode == "冲高五点回落" and high is not None:
            pct_chg = (high / prev_close.replace(0, NOT_A_NUMBER) - 1) * 100
            signal[(pct_chg > 5) & (close < high)] = 1
        elif self.mode == "高开低走" and open_p is not None:
            signal[(open_p > prev_close) & (close < open_p)] = 1
        elif self.mode == "平开低走" and open_p is not None:
            signal[(open_p == prev_close) & (close < open_p)] = 1
        elif self.mode == "开盘低于昨收" and open_p is not None:
            signal[open_p < prev_close] = 1
        elif self.mode == "打开涨停":
            pct_chg = (close / prev_close.replace(0, NOT_A_NUMBER) - 1) * 100
            signal[pct_chg > 9.5] = 1
        elif self.mode == "打开涨停回落":
            pct_chg = (close / prev_close.replace(0, NOT_A_NUMBER) - 1) * 100
            signal[(pct_chg > 9.5) & (close < high)] = 1

        return signal.fillna(0).astype(int)


class MarketTimingV3_1p5(TimingSignalAdapter):
    """大盘择时_v3-1p5 — v3变体，含分批入场(scalein)和离场(override)信号。

    原版 scalein_targets: [0.50, 0.76, 0.90, 0.97, 1.0] (疯狂版)
    """

    def __init__(
        self, entry_threshold: float = 0.8, exit_threshold: float = 1.4
    ) -> None:
        super().__init__(name="market_timing_v3_1p5")
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        open_col = kwargs.get("open_col", "开盘至今涨幅")
        ovn_col = kwargs.get("ovn_col", "隔夜涨跌幅")

        if open_col not in df.columns or ovn_col not in df.columns:
            return pd.Series(dtype=int)

        open_decl = df.groupby("交易日期")[open_col].apply(lambda x: (x < 0).mean())
        ovn_decl = df.groupby("交易日期")[ovn_col].apply(lambda x: (x < 0).mean())
        decl_ratio = open_decl + ovn_decl

        return (decl_ratio <= self.entry_threshold).astype(int)


class MarketTimingV3_2(TimingSignalAdapter):
    """大盘择时_v3-2 — 二级信号体系（下跌比例 + 强度rank 双阈值）。"""

    def __init__(
        self, decl_threshold: float = 0.8, strength_threshold: float = 1.1
    ) -> None:
        super().__init__(name="market_timing_v3_2")
        self.decl_threshold = decl_threshold
        self.strength_threshold = strength_threshold

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        open_col = kwargs.get("open_col", "开盘至今涨幅")
        ovn_col = kwargs.get("ovn_col", "隔夜涨跌幅")

        if open_col not in df.columns or ovn_col not in df.columns:
            return pd.Series(dtype=int)

        # 一级：下跌比例信号
        open_decl = df.groupby("交易日期")[open_col].apply(lambda x: (x < 0).mean())
        ovn_decl = df.groupby("交易日期")[ovn_col].apply(lambda x: (x < 0).mean())
        signal1 = (open_decl + ovn_decl < self.decl_threshold).astype(int)

        # 二级：强度 rank 信号
        open_strength = df.groupby("交易日期")[open_col].mean()
        ovn_strength = df.groupby("交易日期")[ovn_col].mean()
        open_rank = open_strength.rolling(60, min_periods=1).rank(pct=True)
        ovn_rank = ovn_strength.rolling(60, min_periods=1).rank(pct=True)
        signal2 = (open_rank + ovn_rank > self.strength_threshold).astype(int)

        return (signal1 & signal2).astype(int).fillna(0)


class MarketTimingV2(TimingSignalAdapter):
    """大盘择时 v2 — 两级信号体系（下跌比例 + 强度rank，OR 逻辑）。

    大盘择时信号 v2。

    与 v3-2 的区别：v2 使用 OR 逻辑（一级或二级满足即开仓），
    v3-2 使用 AND 逻辑（两级同时满足才开仓）。
    """

    def __init__(
        self, decl_threshold: float = 0.8, strength_threshold: float = 1.1
    ) -> None:
        super().__init__(name="market_timing_v2")
        self.decl_threshold = decl_threshold
        self.strength_threshold = strength_threshold

    def compute_timing(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        open_col = kwargs.get("open_col", "开盘至今涨幅")
        ovn_col = kwargs.get("ovn_col", "隔夜涨跌幅")

        if open_col not in df.columns or ovn_col not in df.columns:
            return pd.Series(dtype=int)

        # 一级：下跌比例信号（同 v1.5）
        open_decl = df.groupby("交易日期")[open_col].apply(lambda x: (x < 0).mean())
        ovn_decl = df.groupby("交易日期")[ovn_col].apply(lambda x: (x < 0).mean())
        signal1 = (open_decl + ovn_decl < self.decl_threshold).astype(int)

        # 二级：强度 rank 信号
        open_strength = df.groupby("交易日期")[open_col].mean()
        ovn_strength = df.groupby("交易日期")[ovn_col].mean()
        open_rank = open_strength.rolling(60, min_periods=1).rank(pct=True)
        ovn_rank = ovn_strength.rolling(60, min_periods=1).rank(pct=True)
        signal2 = (open_rank + ovn_rank > self.strength_threshold).astype(int)

        # OR 逻辑：一级满足则开仓，否则看二级
        result = signal1.combine(
            signal2, lambda s1, s2: 1 if s1 == 1 else (1 if s2 == 1 else 0)
        )
        return result.fillna(0).astype(int)
