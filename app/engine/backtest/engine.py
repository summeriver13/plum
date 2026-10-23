"""回测引擎 — 基于事件驱动的指令式架构模拟。

退市保护：
    引擎集成 DelistingRegistry 和 UniverseFilter 以防止：
    1. 买入已退市股票（由 UniverseFilter 拦截）
    2. 僵尸持仓（每日调用 Portfolio.liquidate_delisted()）
    3. 幽灵信号（过滤掉已退市股票的信号）

精度约定：
    所有价格/金额字段使用 Decimal 存储和计算。
"""

from __future__ import annotations

import bisect
import logging
import time as time_module
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict

from app.data.schema import Bar, Order, OrderStatus, TradeRecord
from app.engine.backtest.models import (
    BacktestResult,
    EquityPoint,
    Signal,
    SignalDirection,
    StrategyAttribution,
)
from app.engine.backtest.parking import (
    DEFAULT_FEE_BUFFER,
    DEFAULT_MIN_TRADE_RATIO,
    ETF_LOT_SIZE,
    ParkingBook,
    ParkingTrade,
    floor_lots,
    load_parking_bars,
    plan_liquidation,
    plan_rebalance,
)
from app.engine.backtest.portfolio import Portfolio, Position
from app.engine.backtest.scenarios.commission import AShareCommissionModel
from app.engine.backtest.scenarios.impact import (
    DEFAULT_PARTICIPATION_LIMIT,
    MAX_LIMIT_DEVIATION,
    ImpactModel,
    daily_turnover,
    max_participating_quantity,
    no_impact,
)
from app.engine.backtest.scenarios.limit_up_down import LimitUpDownGuard, MatchMode
from app.engine.backtest.scenarios.slippage import SlippageModel, no_slippage

if TYPE_CHECKING:
    from app.data.delisting_registry import DelistingRegistry
    from app.engine.backtest.executor import Executor
    from app.engine.backtest.pipeline import FilterContext
    from app.engine.backtest.scenarios.universe_filter import FilteredSignal
    from app.engine.backtest.sizers import SizerContext
    from app.rule.corporate.actions import CorporateAction, CorporateActionProcessor
    from app.rule.trading.calculator import TradingRules

logger = logging.getLogger(__name__)


def _as_float(value: Decimal | int | float | str) -> float:
    """安全转换为 float（结果元数据序列化用）。"""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法转换为浮点数: {value!r}") from exc


def _as_decimal(value: object, default: Decimal = Decimal("1")) -> Decimal:
    """安全转换为 Decimal（复权因子用：None/NaN/非法值取默认）。"""
    if value is None:
        return default
    try:
        result = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return default
    return default if result.is_nan() else result


def _as_int(value: Decimal | int | float | str) -> int:
    """安全转换为 int（股数取整用）。"""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法转换为整数: {value!r}") from exc


class Strategy(Protocol):
    def on_bar(
        self,
        ts: datetime,
        bars: dict[str, Bar],
        portfolio: Portfolio,
    ) -> dict[str, int] | list[Signal]:
        """每个交易日给出目标持仓（dict）或原始信号（list）。"""
        ...

    def flush_state(self) -> None:
        """刷新策略内部状态（可选实现）。"""
        ...


class MarketCatalog(Protocol):
    """行情目录接口（注入的 CatalogManager）。"""

    def load_bars(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        **kwargs: object,
    ) -> list[Bar]:
        """加载指定标的的日线序列。"""
        ...

    def get_stock_names(self) -> dict[str, str]:
        """标的名称表（ST / 退市判定用）。"""
        ...


class SignalPipeline(Protocol):
    """信号过滤管线接口（可插拔过滤器链）。"""

    def apply(self, signals: list[Signal], context: FilterContext) -> list[Signal]:
        """按过滤器链变换信号。"""
        ...


class PositionSizer(Protocol):
    """仓位分配器接口（信号 → 目标持仓）。"""

    def size(self, signals: list[Signal], context: SizerContext) -> dict[str, int]:
        """把信号转换为目标持仓数量。"""
        ...


class OrderFilter(Protocol):
    """标的池过滤器接口（退市 / ST 拦截）。"""

    def filter_orders(
        self,
        target_positions: dict[str, int],
        current_bars: dict[str, Bar],
        current_ts: datetime,
    ) -> tuple[dict[str, int], list[FilteredSignal]]:
        """过滤目标持仓并返回被拦截清单。"""
        ...


class BoardFlags(TypedDict, total=False):
    """涨跌停守卫所需的板块标记。"""

    is_st: bool
    is_kcb: bool
    is_chinext: bool
    limit_rate: Decimal | None


SHARE_ADJUSTMENT_TOLERANCE = 0.005

ADJ_FACTOR_EPSILON = 1e-6
"""复权因子变化的最小判定阈值（因子只在真实公司行为日跳变，无需大容差挡噪声）。"""
"""参考价与昨收比率偏差低于该值时视为无除权：覆盖 0.001 最小报价单位的舍入，
低于 ETF 现金分红常见幅度（1-3%）。"""


@dataclass
class Engine:
    portfolio: Portfolio
    catalog: MarketCatalog
    commission: AShareCommissionModel
    slippage: SlippageModel = field(default_factory=no_slippage)
    # 流动性约束：单笔 ≤ 当日成交额 × participation_limit（≤0 关闭）；impact 为平方根冲击
    participation_limit: Decimal = DEFAULT_PARTICIPATION_LIMIT
    impact: ImpactModel = field(default_factory=no_impact)
    cash_guard: object = field(default_factory=lambda: None)
    limit_guard: LimitUpDownGuard = field(default_factory=LimitUpDownGuard)
    match_mode: MatchMode = field(default=MatchMode.CLOSE)
    executor: Executor | None = None
    kelly_multiplier: Decimal = Decimal("1.0")
    # 闲置资金停泊（国债逆回购 / 货基口径）：年化收益率 ≤ 0 关闭
    cash_yield: float = 0.0
    park_ratio: float = 1.0
    # 闲置资金停泊（真实标的）：代码为空 = 关闭；与 cash_yield 互斥（启用时利率口径置 0）
    parking_symbol: str | None = None
    parking_lot_size: int = ETF_LOT_SIZE
    parking_min_trade_ratio: float = DEFAULT_MIN_TRADE_RATIO
    parking_bars: list[Bar] = field(default_factory=list)
    """显式注入的停泊标的日线（跨副本停泊用）；非空时优先于 catalog。"""
    parking_market_root: str | Path | None = None
    """catalog 里找不到停泊标的时的备用行情副本根目录（如 data/market/etf）。"""
    parking_log: list[ParkingTrade] = field(default_factory=list)
    _parking_bars: dict[datetime, Bar] = field(
        default_factory=dict, init=False, repr=False
    )
    max_positions: int = 8
    equity_curve: list[EquityPoint] = field(default_factory=list)
    trade_log: list[TradeRecord] = field(default_factory=list)
    order_log: list[Order] = field(default_factory=list)
    _pending_orders: list[Order] = field(default_factory=list)
    _strategy_max_pos: int | None = field(default=None, repr=False)

    # 成交价自检：裁剪发生率（裁剪 = 成交价被涨跌停带收紧），随结果元数据输出
    _exec_price_calls: int = field(default=0, init=False, repr=False)
    _clip_count: int = field(default=0, init=False, repr=False)
    _no_limit_calls: int = field(default=0, init=False, repr=False)
    _clip_notional: Decimal = field(
        default_factory=lambda: Decimal("0"), init=False, repr=False
    )

    # 退市保护
    delisting_registry: DelistingRegistry | None = field(default=None, repr=False)
    universe_filter: OrderFilter | None = field(default=None, repr=False)
    enable_delisting_protection: bool = True
    # 退市清算的残值比例：1.0 = 按最后收盘价清算（默认），0.0 = 归零（压力测试）
    delisting_residual_ratio: float = 1.0
    # 退市清算统计（写入结果 metadata，供报告量化“僵尸持仓”修复效果）
    delisting_liquidation_count: int = 0
    delisting_loss_total: Decimal = Decimal("0")
    exclude_st: bool = True  # 默认过滤 ST，有需求显式关闭

    # 拆单
    enable_order_splitting: bool = False
    split_min_value: Decimal = Decimal("50000")  # 低于此金额不拆单

    # 事件处理
    action_processor: CorporateActionProcessor | None = field(default=None, repr=False)
    share_adjustment: bool = True
    """除权复权：按复权因子变化调整持仓数量（默认开启，正确性内建）。

    旧批次（v8-v10 股票）为复现已发布数字，在脚本里显式传 False 固定旧口径。
    """
    """是否按除权除息参考价复权持仓数量（送转 / 份额拆分与合并）。

    默认关闭：既有股票批次（v8-v10）未含该修正，开启会改变其口径；
    ETF 批次必须开启（份额拆分会产生 5 倍以上的原始价跳变）。
    """
    _pending_actions: dict[str, list[CorporateAction]] = field(
        default_factory=dict, repr=False
    )

    # 因子预计算缓存
    factor_store: object | None = field(default=None, repr=False)

    # 信号管线（可插拔过滤器链）
    pipeline: SignalPipeline | None = field(default=None, repr=False)

    # 仓位分配器（可替换的信号→股数策略）
    sizer: PositionSizer | None = field(default=None, repr=False)

    # 多策略归因追踪
    track_attribution: bool = False
    _attribution: dict[str, StrategyAttribution] = field(
        default_factory=dict, init=False, repr=False
    )
    _symbol_strategy: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    # A 股规则
    rules: TradingRules | None = field(default=None, repr=False)

    def run(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        strategy: Strategy,
        liquidate: bool = True,
        progress_callback: Callable[[float, dict[str, object]], None] | None = None,
    ) -> BacktestResult:
        self._exec_price_calls = 0
        self._clip_count = 0
        self._no_limit_calls = 0
        self._clip_notional = Decimal("0")
        # 闲置资金停泊参数下沉到账本（利息在日循环内按日计提）
        self.portfolio.cash_yield = self.cash_yield
        self.portfolio.park_ratio = self.park_ratio
        self._setup_parking()

        from app.data.delisting_registry import get_registry
        from app.engine.backtest.scenarios.universe_filter import UniverseFilter

        if self.enable_delisting_protection:
            if self.delisting_registry is None:
                self.delisting_registry = get_registry()
            if self.universe_filter is None:
                self.universe_filter = UniverseFilter(
                    self.delisting_registry, exclude_st=self.exclude_st
                )

        from app.rule.corporate.actions import CorporateActionProcessor

        if self.action_processor is None:
            self.action_processor = CorporateActionProcessor()

        if self.share_adjustment:
            self._register_default_processors()

        if self.rules is None:
            from app.rule.trading.calculator import ASHARE_RULES

            self.rules = ASHARE_RULES

        warmup_days = getattr(strategy, "window", 0)
        warmup_start = start
        if warmup_days > 0:
            warmup_start = start - timedelta(days=warmup_days * 3)

        active_symbols = self._filter_active_symbols(symbols, start)
        logger.info(
            f"Universe pre-filter: {len(symbols)} → {len(active_symbols)} active "
            f"(removed {len(symbols) - len(active_symbols)} delisted before start)"
        )

        all_data: dict[str, list[Bar]] = {}
        if len(active_symbols) <= 1:
            for s in active_symbols:
                bars = self.catalog.load_bars(s, start=warmup_start, end=end)
                if bars:
                    all_data[s] = bars
        else:
            max_workers = min(8, len(active_symbols))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        lambda s=s: self.catalog.load_bars(
                            s, start=warmup_start, end=end
                        )
                    ): s
                    for s in active_symbols
                }
                for future in as_completed(futures):
                    s = futures[future]
                    try:
                        bars = future.result()
                        if bars:
                            all_data[s] = bars
                    except Exception as e:
                        logger.warning(f"Failed to load bars for {s}: {e}")

        all_timestamps = self._build_timeline(all_data)
        if not all_timestamps:
            return BacktestResult(
                symbol=",".join(symbols),
                start_date=start,
                end_date=end,
                initial_cash=self.portfolio.initial_cash,
                trade_log=self.trade_log,
                order_log=self.order_log,
                equity_curve=[],
                metadata={
                    "delisting_liquidations": 0,
                    "delisting_loss_total": 0.0,
                    "exec_sanity": self._exec_sanity_metadata(),
                },
            )

        all_timestamps = [ts for ts in all_timestamps if start <= ts <= end]

        indices = dict.fromkeys(all_data, 0)
        self._strategy_max_pos = getattr(strategy, "max_positions", None)

        indicator_calculator = getattr(strategy, "indicator_calculator", None)
        if indicator_calculator is not None:
            from app.factors.precomputed import build_precomputed_cache

            cache = build_precomputed_cache(
                all_data,
                ema_fast_period=indicator_calculator.ema_fast_period,
                ema_slow_period=indicator_calculator.ema_slow_period,
                ma_trend_period=indicator_calculator.ma_trend_period,
                atr_period=indicator_calculator.atr_period,
                adx_period=indicator_calculator.adx_period,
                bb_period=indicator_calculator.bb_period,
                bb_std=indicator_calculator.bb_std,
                rsi_period=indicator_calculator.rsi_period,
            )
            indicator_calculator.attach_precomputed(cache)

        # 因子预计算缓存（通过 FactorStore 接口）
        factor_scores: dict[datetime, dict[str, float]] = {}
        if self.factor_store is not None:
            factor_store = self.factor_store
            for ts in all_timestamps:
                if start <= ts <= end:
                    scores = factor_store.score_at(ts)  # type: ignore[union-attr]
                    if scores:
                        factor_scores[ts] = scores

        total_days = len(all_timestamps)
        start_time = time_module.perf_counter()
        last_report_time = start_time
        rolling_times: list[float] = []

        watched_attr = getattr(strategy, "watched_symbols", None)
        watched_symbols: frozenset[str] | None = (
            frozenset(watched_attr) if watched_attr else None
        )

        if self.parking_symbol:
            self._load_parking_bars(start, end)

        for day_index, ts in enumerate(all_timestamps):
            current_bars = self._get_current_bars(all_data, indices, ts)

            # 退市保护：清仓已退市的持仓
            if self.enable_delisting_protection and self.delisting_registry:
                delisted_losses = self.portfolio.liquidate_delisted(
                    ts,
                    self.delisting_registry,
                    residual_value_ratio=self.delisting_residual_ratio,
                )
                if delisted_losses:
                    syms = [loss.symbol for loss in delisted_losses]
                    self.delisting_liquidation_count += len(delisted_losses)
                    self.delisting_loss_total += sum(
                        (loss.loss for loss in delisted_losses), Decimal("0")
                    )
                    logger.warning(
                        f"[{ts.date()}] Liquidated {len(delisted_losses)} "
                        f"delisted positions: {syms}"
                    )

            # ST 股票保护：清仓已变为 ST 的持仓
            st_stocks: set[str] = set()
            if self.exclude_st:
                stock_names = self.catalog.get_stock_names()
                st_stocks = {
                    sym
                    for sym, name in stock_names.items()
                    if "ST" in name.upper() or "退市" in name
                }
                if st_stocks:
                    st_losses = self.portfolio.liquidate_st(st_stocks, ts)
                    if st_losses:
                        syms = [loss.symbol for loss in st_losses]
                        logger.warning(
                            f"[{ts.date()}] Liquidated {len(st_losses)} "
                            f"ST positions: {syms}"
                        )

            # 除权除息/份额拆分：先复权持仓数量，再刷新价格
            if self.share_adjustment:
                adjusted = self._apply_share_adjustments(ts, current_bars)
                if adjusted:
                    logger.info(
                        f"[{ts.date()}] Share adjustment applied to {adjusted} positions"
                    )

            # 更新持仓价格
            for sym, bar in current_bars.items():
                self.portfolio.update_price(sym, bar.close)

            # 停泊标的估值（独立行情，不进入策略可见的 bars）
            parking_bar = self._parking_bars.get(ts) if self.parking_symbol else None
            if parking_bar is not None:
                self.portfolio.update_parking_price(parking_bar.close)

            # 闲置资金停泊收益：按当日可用资金计提，入账后当日仓位规模随之放大
            day_interest = self.portfolio.accrue_cash_interest()

            # 处理当日的除权除息等事件
            if self.action_processor is not None:
                self._process_corporate_actions(ts, current_bars)

            # 聚焦 K 线优化
            if watched_symbols is not None:
                portfolio_keys = self.portfolio.positions.keys()
                focused_symbols = watched_symbols | frozenset(portfolio_keys)
                focused_bars = {
                    s: bar for s, bar in current_bars.items() if s in focused_symbols
                }
            else:
                focused_bars = current_bars

            # 从策略获取信号
            raw_signals = strategy.on_bar(ts, focused_bars, self.portfolio)

            if hasattr(strategy, "flush_state"):
                strategy.flush_state()

            # ── 信号管线与仓位分配 ────────────────────────
            # 策略契约允许返回目标持仓字典（见 Strategy.on_bar）：管线与分配器
            # 只吃信号列表，字典本身就是目标持仓
            if isinstance(raw_signals, list):
                target_positions = self._signal_target_positions(
                    raw_signals, current_bars, ts
                )
            else:
                target_positions = dict(raw_signals)

            # ── 策略归因追踪 ─────────────────────────────
            if self.track_attribution and isinstance(raw_signals, list):
                self._update_attribution(raw_signals, current_bars)
                # 维护 symbol → strategy_id 映射（用于 NAV 归因）
                for sig in raw_signals:
                    if sig.direction == SignalDirection.LONG:
                        sid = sig.metadata.get("strategy_id", "__unknown__")
                        self._symbol_strategy[sig.symbol] = sid

            # 过滤退市股票的目标持仓
            if self.enable_delisting_protection and self.universe_filter:
                blocked: list[FilteredSignal]
                target_positions, blocked = self.universe_filter.filter_orders(
                    target_positions, current_bars, ts
                )
                if blocked:
                    logger.debug(
                        f"[{ts.date()}] Blocked {len(blocked)} signals: "
                        f"{[(b.signal.symbol, b.reason) for b in blocked]}"
                    )

            if self.executor is not None:
                target_positions = self._apply_executor_filter(
                    target_positions, current_bars
                )

            orders = self._create_orders(ts, target_positions, current_bars)

            # 停泊赎回：为当日买入腾出现金（ETF T+0，先于订单执行）
            if parking_bar is not None:
                self._fund_orders_from_parking(orders, current_bars, parking_bar)

            # 拆单（大单拆成 TWAP 小单，小资金策略自动跳过）
            if self.enable_order_splitting:
                from app.engine.backtest.order_splitter import split_orders

                split_orders_list = []
                for order in orders:
                    bar = current_bars.get(order.symbol)
                    if bar and self._needs_splitting(order, bar.close):
                        split_orders_list.extend(split_orders([order], bar.close))
                    else:
                        split_orders_list.append(order)
                orders = split_orders_list

            for order in orders:
                if self.match_mode == MatchMode.NEXT_OPEN:
                    self._add_pending_order(order)
                else:
                    self._execute_single_order(order, current_bars.get(order.symbol))

            self._process_pending_orders(ts, current_bars)

            # 停泊再平衡：把闲置现金按 park_ratio 补入停泊仓（阈值触发）
            if parking_bar is not None:
                self._rebalance_parking(parking_bar)

            self.portfolio.on_day_end()

            positions_value = sum(
                (p.market_value for p in self.portfolio.positions.values()),
                Decimal("0"),
            )
            positions_detail = {
                s: p.quantity for s, p in self.portfolio.positions.items()
            }

            # ── 子策略 NAV 归因 ──────────────────────────
            if self.track_attribution:
                strategy_values: dict[str, Decimal] = {}
                for sym, pos in self.portfolio.positions.items():
                    sid = self._symbol_strategy.get(sym, "__unknown__")
                    strategy_values[sid] = (
                        strategy_values.get(sid, Decimal("0")) + pos.market_value
                    )
                for sid in list(self._attribution.keys()):
                    self._attribution[sid].nav_series.append(
                        _as_float(strategy_values.get(sid, Decimal("0")))
                    )
                for sid, val in strategy_values.items():
                    if sid not in self._attribution:
                        self._attribution[sid] = StrategyAttribution(strategy_id=sid)
                        self._attribution[sid].nav_series = [0.0] * len(
                            self.equity_curve
                        )
                        self._attribution[sid].nav_series.append(_as_float(val))

            equity_point = EquityPoint(
                timestamp=ts,
                total_value=self.portfolio.total_value,
                cash=self.portfolio.cash,
                frozen_cash=self.portfolio.frozen_cash,
                positions_value=positions_value,
                positions_detail=positions_detail,
                cash_interest=day_interest,
                parking_value=self.portfolio.parking_value,
            )
            self.equity_curve.append(equity_point)

            # 进度报告
            if progress_callback is not None:
                now = time_module.perf_counter()
                elapsed_total = now - start_time
                report_now = (
                    day_index % 5 == 0
                    or day_index == total_days - 1
                    or now - last_report_time >= 1.0
                )
                if report_now:
                    last_report_time = now
                    rolling_times.append(elapsed_total / (day_index + 1))
                    if len(rolling_times) > 10:
                        rolling_times.pop(0)
                    avg_per_day = sum(rolling_times) / len(rolling_times)
                    remaining_days = total_days - day_index - 1
                    eta_seconds = (
                        avg_per_day * remaining_days if remaining_days > 0 else None
                    )
                    progress_callback(
                        (day_index + 1) / total_days,
                        {
                            "current_date": ts,
                            "total_days": total_days,
                            "elapsed_days": day_index + 1,
                            "trades_total": len(self.trade_log),
                            "portfolio_value": self.portfolio.total_value,
                            "eta_seconds": eta_seconds,
                        },
                    )

        if liquidate:
            self._liquidate_all(all_data, indices, all_timestamps[-1])
            final_positions_value = sum(
                (p.market_value for p in self.portfolio.positions.values()),
                Decimal("0"),
            )
            final_detail = {s: p.quantity for s, p in self.portfolio.positions.items()}
            final_equity_point = EquityPoint(
                timestamp=all_timestamps[-1],
                total_value=self.portfolio.total_value,
                cash=self.portfolio.cash,
                frozen_cash=Decimal("0"),
                positions_value=final_positions_value,
                positions_detail=final_detail,
                cash_interest=self.equity_curve[-1].cash_interest,
                parking_value=self.portfolio.parking_value,
            )
            self.equity_curve[-1] = final_equity_point

        return BacktestResult(
            symbol=",".join(symbols),
            start_date=start,
            end_date=end,
            initial_cash=self.portfolio.initial_cash,
            trade_log=self.trade_log,
            order_log=self.order_log,
            equity_curve=self.equity_curve,
            parking_log=self.parking_log,
            attribution=dict(self._attribution),
            metadata={
                "match_mode": self.match_mode.value,
                "delisting_liquidations": self.delisting_liquidation_count,
                "delisting_loss_total": _as_float(self.delisting_loss_total),
                "exec_sanity": self._exec_sanity_metadata(),
                "cash_interest_total": _as_float(self.portfolio.cash_interest_total),
                "cash_yield": self.cash_yield,
                "park_ratio": self.park_ratio,
                "parking_symbol": self.parking_symbol or "",
                "parking_final_value": _as_float(self.portfolio.parking_value),
                "parking_realized_pnl": _as_float(self.portfolio.parking.realized_pnl),
                "parking_trade_count": len(self.parking_log),
                "parking_turnover": _as_float(self._parking_turnover()),
            },
        )

    # ─── 闲置资金停泊（真实标的口径）────────────────────────────

    def _setup_parking(self) -> None:
        """初始化停泊口径：与利率口径互斥，重置账本与行情缓存。"""
        self.parking_log = []
        self._parking_bars = {}
        if not self.parking_symbol:
            return
        if self.cash_yield > 0:
            logger.warning(
                "停泊标的 %s 已启用，cash_yield=%s 置 0（避免与标的收益双算）",
                self.parking_symbol,
                self.cash_yield,
            )
            self.cash_yield = 0.0
            self.portfolio.cash_yield = 0.0
        self.portfolio.parking = ParkingBook(symbol=self.parking_symbol)

    def _load_parking_bars(self, start: datetime, end: datetime) -> None:
        """加载停泊标的行情（独立于策略可见 bars，不污染标的池）。

        优先级：显式注入的 ``parking_bars`` → 策略 catalog → ``parking_market_root``
        指定的副本（股票策略停泊 ETF 时用）。
        """
        symbol = self.parking_symbol
        if not symbol:
            return
        if self.parking_bars:
            self._parking_bars = {bar.timestamp: bar for bar in self.parking_bars}
            return

        bars: list[Bar] = []
        try:
            bars = list(self.catalog.load_bars(symbol, start=start, end=end))
        except Exception:  # noqa: BLE001 - 行情加载失败时降级为不停泊
            logger.warning(
                "停泊标的 %s 行情加载失败，停泊不生效", symbol, exc_info=True
            )
        if not bars and self.parking_market_root:
            logger.info(
                "停泊标的 %s 不在策略行情副本，改从 %s 读取",
                symbol,
                self.parking_market_root,
            )
            bars = load_parking_bars(symbol, start, end, self.parking_market_root)
        self._parking_bars = {bar.timestamp: bar for bar in bars}
        if not self._parking_bars:
            logger.warning(
                "停泊标的 %s 在 %s~%s 无行情，停泊不生效",
                symbol,
                start.date(),
                end.date(),
            )

    def _fund_orders_from_parking(
        self, orders: list[Order], current_bars: dict[str, Bar], parking_bar: Bar
    ) -> None:
        """为当日买入赎回停泊仓（挂单买入已冻结现金，无需腾挪）。"""
        if self.match_mode == MatchMode.NEXT_OPEN:
            return
        need = Decimal("0")
        for order in orders:
            bar = current_bars.get(order.symbol)
            if bar is None:
                continue
            notional = Decimal(abs(order.qty)) * bar.close
            need += notional if order.is_buy else -notional
        shortfall = (
            need * (Decimal("1") + DEFAULT_FEE_BUFFER) - self.portfolio.available_cash
        )
        trade = plan_liquidation(
            symbol=self.parking_symbol or "",
            price=parking_bar.close,
            shortfall=shortfall,
            current_quantity=self.portfolio.parking_quantity,
            lot_size=self.parking_lot_size,
        )
        if trade is not None:
            self._execute_parking_trade(trade, parking_bar)

    def _rebalance_parking(self, parking_bar: Bar) -> None:
        """把闲置现金按 park_ratio 补入停泊仓（阈值触发，避免每日碎单）。"""
        trade = plan_rebalance(
            symbol=self.parking_symbol or "",
            price=parking_bar.close,
            available_cash=self.portfolio.available_cash,
            park_ratio=self.park_ratio,
            nav=self.portfolio.total_value,
            lot_size=self.parking_lot_size,
            min_trade_ratio=self.parking_min_trade_ratio,
        )
        if trade is not None:
            self._execute_parking_trade(trade, parking_bar)

    def _execute_parking_trade(self, trade: ParkingTrade, bar: Bar) -> None:
        """按主仓同口径摩擦执行停泊成交（参与率/冲击/涨跌停带一致）。"""
        qty = abs(trade.quantity)
        is_buy = trade.quantity > 0
        exec_price = self._get_exec_price(bar, is_buy, qty)
        capped = max_participating_quantity(
            exec_price, bar.amount, self.participation_limit, lot=self.parking_lot_size
        )
        if capped is not None:
            qty = min(qty, capped)
        if is_buy:
            qty = min(
                qty,
                self._affordable_parking_quantity(
                    exec_price, self.portfolio.available_cash, qty
                ),
            )
        if qty < self.parking_lot_size:
            return
        fee = self.commission.calculate(exec_price, qty, is_buy).total
        signed = qty if is_buy else -qty
        self.portfolio.apply_parking_trade(signed, exec_price, fee)
        self.parking_log.append(
            ParkingTrade(
                symbol=trade.symbol,
                quantity=signed,
                price=exec_price,
                reason=trade.reason,
                fee=fee,
                timestamp=bar.timestamp,
            )
        )

    def _affordable_parking_quantity(
        self, price: Decimal, cash: Decimal, want: int
    ) -> int:
        """按可用现金与佣金裁剪买入手数（预留费用缓冲，再按实际佣金校验）。"""
        lot = self.parking_lot_size
        lot_cost = Decimal(lot) * price
        if lot_cost <= Decimal("0"):
            return 0
        budget = cash / (Decimal("1") + DEFAULT_FEE_BUFFER)
        qty = min(want, floor_lots(budget, lot_cost) * lot)
        while qty > 0:
            fee = self.commission.calculate(price, qty, True).total
            if Decimal(qty) * price + fee <= cash:
                return qty
            qty -= lot
        return 0

    def _parking_turnover(self) -> Decimal:
        """停泊累计成交额（元）。"""
        return sum((trade.notional for trade in self.parking_log), Decimal("0"))

    def _update_attribution(
        self,
        signals: list[Signal],
        bars: dict[str, Bar],
    ) -> None:
        """更新子策略归因统计。

        统计每个子策略的 LONG/FLAT 信号数量。
        NAV 序列在 equity 记录阶段更新。
        """
        for sig in signals:
            sid = sig.metadata.get("strategy_id", "__unknown__")
            if sid not in self._attribution:
                self._attribution[sid] = StrategyAttribution(strategy_id=sid)
            attr = self._attribution[sid]
            if sig.direction == SignalDirection.LONG:
                attr.long_signal_count += 1
            elif sig.direction == SignalDirection.FLAT:
                attr.flat_signal_count += 1

    def _signal_target_positions(
        self,
        signals: list[Signal],
        current_bars: dict[str, Bar],
        ts: datetime,
    ) -> dict[str, int]:
        """信号列表 → 目标持仓：先过信号管线，再过仓位分配器（缺省用内置换算）。"""
        if self.pipeline is not None:
            from app.engine.backtest.pipeline import FilterContext

            ctx = FilterContext(ts=ts, bars=current_bars, portfolio=self.portfolio)
            signals = self.pipeline.apply(signals, ctx)
        if self.sizer is None:
            # 回退到内置转换
            return self._convert_signals(
                signals, current_bars, self.portfolio.total_value
            )

        from app.engine.backtest.sizers import SizerContext

        prices = {sym: bar.close for sym, bar in current_bars.items()}
        sizer_ctx = SizerContext(
            portfolio_value=self.portfolio.total_value,
            current_holdings={
                s: p.quantity for s, p in self.portfolio.positions.items()
            },
            prices=prices,
            max_positions=self._strategy_max_pos or self.max_positions,
            rules=self.rules,
            kelly_multiplier=self.kelly_multiplier,
        )
        return self.sizer.size(signals, sizer_ctx)

    def _convert_signals(
        self,
        raw,
        current_bars: dict[str, Bar],
        portfolio_value: Decimal,
    ) -> dict[str, int]:
        if isinstance(raw, dict):
            return raw  # type: ignore[return-value]
        if isinstance(raw, list):
            signals: list[Signal] = raw
            if not signals:
                return {}
            result: dict[str, int] = {}
            current_holdings = {
                s: p.quantity for s, p in self.portfolio.positions.items()
            }
            allocated = 0

            # 当日卖出预计回款：A 股卖出资金 T+0 可用于买入，但卖出要到
            # 执行阶段才落账。若只用 available_cash 判断，调仓日的新买
            # 只能靠前一日残余现金（满仓时仅 ~2%）→ 大量买入被跳过，
            # 仓位长期被迫失衡（现金拖累）。此处把待卖出回款计入预算，
            # 执行器仍会按真实现金做最终约束。
            pending_proceeds = Decimal("0")
            for signal in signals:
                if (
                    signal.direction == SignalDirection.FLAT
                    and current_holdings.get(signal.symbol, 0) > 0
                ):
                    flat_bar = current_bars.get(signal.symbol)
                    if flat_bar is not None:
                        pending_proceeds += (
                            Decimal(current_holdings[signal.symbol]) * flat_bar.close
                        )
            cash_budget = self.portfolio.available_cash + pending_proceeds

            for signal in signals:
                symbol = signal.symbol
                bar = current_bars.get(symbol)
                if bar is None:
                    continue
                price = bar.close

                current_qty = current_holdings.get(symbol, 0)

                if signal.direction == SignalDirection.LONG and current_qty == 0:
                    max_pos = (
                        self._strategy_max_pos
                        if self._strategy_max_pos is not None
                        else self.max_positions
                    )
                    if allocated >= max_pos:
                        continue
                    per_position = (
                        portfolio_value
                        * Decimal(str(signal.strength))
                        * self.kelly_multiplier
                    ) / Decimal(max_pos)
                    qty_float = per_position / price
                    qty = _as_int(qty_float)
                    if self.rules is not None:
                        qty = self.rules.floor_to_lot(symbol, qty)
                    else:
                        qty = (qty // 100) * 100
                    if self.rules is not None:
                        is_valid = self.rules.is_valid_lot(symbol, qty)
                    else:
                        is_valid = qty >= 100
                    if is_valid:
                        est_cost = Decimal(qty) * price
                        if cash_budget < est_cost:
                            logger.debug(
                                f"Cash Guard: {symbol} qty={qty} requires "
                                f"{est_cost} yuan but only "
                                f"{cash_budget} budgeted, skipping"
                            )
                            continue
                        result[symbol] = qty
                        cash_budget -= est_cost
                        allocated += 1
                elif signal.direction == SignalDirection.FLAT and current_qty > 0:
                    result[symbol] = 0
                elif current_qty > 0:
                    result[symbol] = current_qty

            for symbol, qty in current_holdings.items():
                if symbol not in result and qty > 0:
                    result[symbol] = qty

            return result
        return {}

    def _build_timeline(self, all_data: dict[str, list[Bar]]) -> list[datetime]:
        timestamps = set()
        for bars in all_data.values():
            timestamps.update(bar.timestamp for bar in bars)
        return sorted(timestamps)

    def _filter_active_symbols(self, symbols: list[str], start: datetime) -> list[str]:
        """预过滤股票池：只剔除「回测开始前已退市」的标的。

        注意不能用 `universe_filter.is_blocked(s, "BUY", start)`——它会把
        「回测开始后才上市」的标的也判为不可买（未上市），从而整只剔除，
        使这些标的在上市后永远无法进入回测。未上市标的必须保留，
        由买入当日的 UniverseFilter 按日期拦截。
        """
        if not self.enable_delisting_protection or not self.delisting_registry:
            return symbols

        active = []
        for s in symbols:
            lifecycle = self.delisting_registry.get_lifecycle(s)
            if (
                lifecycle is not None
                and lifecycle.last_trading_day is not None
                and lifecycle.last_trading_day < start
            ):
                continue
            active.append(s)

        return active

    def _get_current_bars(
        self,
        all_data: dict[str, list[Bar]],
        indices: dict[str, int],
        ts: datetime,
    ) -> dict[str, Bar]:
        current_bars = {}
        for s, bars in all_data.items():
            idx = indices[s]
            if idx >= len(bars):
                continue

            pos = bisect.bisect_left(bars, ts, lo=idx, key=lambda bar: bar.timestamp)

            if pos < len(bars) and bars[pos].timestamp == ts:
                current_bars[s] = bars[pos]
                indices[s] = pos + 1
        return current_bars

    def _needs_splitting(self, order, price: Decimal) -> bool:
        """判断是否需要拆单：订单金额 ≥ split_min_value 才拆。

        小资金策略（5 万本金以下）自动跳过拆单。
        """
        order_value = Decimal(abs(order.quantity)) * price
        return order_value >= self.split_min_value

    def _create_orders(
        self,
        timestamp: datetime,
        target_positions: dict[str, int],
        current_bars: dict[str, Bar],
    ) -> list[Order]:
        orders = []
        for symbol, target_qty in target_positions.items():
            current_pos = self.portfolio.get_position(symbol)
            current_qty = current_pos.quantity if current_pos else 0
            diff_qty = target_qty - current_qty

            if diff_qty == 0:
                continue

            order = Order(
                order_id=f"ORD-{timestamp.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
                symbol=symbol,
                qty=diff_qty,
                created_ts=timestamp,
                status=OrderStatus.PENDING,
            )
            orders.append(order)
            self.order_log.append(order)

        return orders

    def _add_pending_order(self, order: Order) -> None:
        self._pending_orders.append(order)

        if order.is_buy:
            pos = self.portfolio.positions.get(order.symbol)
            price = (
                pos.last_price
                if pos is not None and pos.last_price > Decimal("0")
                else Decimal("10")
            )
            # 预留 = 名义金额 + 按真实佣金口径算的费用（freeze_cash 只累加 frozen_cash，
            # 不动 cash）；金额记在订单上，成交/撤单时按同一数字精确释放
            estimate = Decimal(abs(order.qty)) * price
            fee = self.commission.calculate(price, abs(order.qty), True).total
            order.frozen_amount = estimate + fee
            self.portfolio.freeze_cash(order.frozen_amount)
            logger.debug(
                f"Order Pending: {order.symbol} BUY {order.qty}, "
                f"frozen {order.frozen_amount}"
            )

    def _process_pending_orders(
        self, timestamp: datetime, current_bars: dict[str, Bar]
    ) -> None:
        if not self._pending_orders or self.match_mode != MatchMode.NEXT_OPEN:
            return

        executed = []
        for order in self._pending_orders:
            bar = current_bars.get(order.symbol)

            if bar is None:
                order.cancel("suspended")
                logger.warning(
                    f"Order Cancelled: {order.symbol} still suspended at {timestamp}. "
                    f"Order from {order.created_ts.date()} expired."
                )
                self._unfreeze_for_order(order)
                executed.append(order)
                continue

            self._execute_single_order(order, bar)
            # 成交路径已清零预留；拒单/挂单未成交时在此释放，避免 frozen_cash 泄漏
            self._unfreeze_for_order(order)
            executed.append(order)

        for order in executed:
            self._pending_orders.remove(order)

    def _unfreeze_for_order(self, order: Order) -> None:
        """释放该买入订单的挂单预留（幂等：释放后清零）。"""
        if order.is_buy and order.frozen_amount > Decimal("0"):
            self.portfolio.unfreeze_cash(order.frozen_amount)
            order.frozen_amount = Decimal("0")

    def _execute_single_order(self, order: Order, bar: Bar | None) -> bool:
        if order.status != OrderStatus.PENDING:
            return False

        if bar is None:
            order.cancel("suspended")
            logger.warning(
                f"Order Cancelled: {order.symbol} suspended. "
                f"Signal from {order.created_ts.date()}."
            )
            return False

        limit_guard_enabled = self.limit_guard is not None
        if limit_guard_enabled:
            if order.is_buy and not self.limit_guard.can_buy(
                bar.close, bar.pre_close, **self._board_flags(bar)
            ):
                order.reject("limit_up")
                logger.warning(
                    "Order Rejected: %s limit up at %s. Buy order from %s.",
                    order.symbol,
                    bar.timestamp,
                    order.created_ts.date(),
                )
                return False
            if order.is_sell and not self.limit_guard.can_sell(
                bar.close, bar.pre_close, **self._board_flags(bar)
            ):
                order.reject("limit_down")
                logger.warning(
                    "Order Rejected: %s limit down at %s. Sell order from %s.",
                    order.symbol,
                    bar.timestamp,
                    order.created_ts.date(),
                )
                return False

        exec_price = self._get_exec_price(bar, order.is_buy, order.qty)
        abs_qty = abs(order.qty)

        capped_qty = self._apply_participation_limit(bar, order, abs_qty)
        if capped_qty == 0:
            order.reject("illiquid")
            logger.warning(
                f"Order Rejected: {order.symbol} participation limit "
                f"(amount={bar.amount}, requested={abs_qty})."
            )
            return False
        if capped_qty < abs_qty:
            logger.debug(
                f"Order Capped: {order.symbol} {abs_qty} -> {capped_qty} "
                f"(amount={bar.amount}, limit={self.participation_limit})."
            )
            abs_qty = capped_qty
            exec_price = self._get_exec_price(bar, order.is_buy, abs_qty)

        if order.is_buy:
            from app.engine.backtest.scenarios.cash_guard import CashGuard

            cash_guard = CashGuard()
            affordable, max_qty = cash_guard.is_affordable(
                exec_price,
                abs_qty,
                self.portfolio.available_cash,
            )
            if not affordable or max_qty == 0:
                order.reject("insufficient_funds")
                logger.warning(
                    f"Order Rejected: {order.symbol} insufficient funds. "
                    f"Requested {abs_qty}, allowed {max_qty}."
                )
                return False
            if max_qty < abs_qty:
                abs_qty = max_qty

        if order.is_sell:
            current_pos = self.portfolio.get_position(order.symbol)
            if current_pos is None:
                order.cancel("no_position")
                logger.warning(f"Order Cancelled: {order.symbol} no position to sell.")
                return False

            available = current_pos.available
            if abs_qty > available:
                logger.warning(
                    f"Order Partial: {order.symbol} requested sell {abs_qty}, "
                    f"but only {available} available (T+1)."
                )
                abs_qty = available
                if abs_qty == 0:
                    order.cancel("t1_restriction")
                    return False

        cost = self.commission.calculate(exec_price, abs_qty, order.is_buy)

        base_price = bar.open if self.match_mode == MatchMode.NEXT_OPEN else bar.close
        slippage_cost = abs(exec_price - base_price) * Decimal(abs_qty)

        self.trade_log.append(
            TradeRecord(
                parent_order_id=order.order_id,
                slippage_cost=slippage_cost,
                timestamp=bar.timestamp,
                symbol=order.symbol,
                quantity=abs_qty,
                price=exec_price,
                cost=cost,
                is_buy=order.is_buy,
            )
        )

        self.portfolio.apply_trade(
            order.symbol,
            abs_qty if order.is_buy else -abs_qty,
            exec_price,
            cost.total,
            bar.timestamp,
        )

        if self.share_adjustment:
            self._sync_entry_state(order.symbol, bar)

        if order.is_buy:
            # 只释放挂单预留：freeze_cash 仅累加 frozen_cash（不动 cash），成交现金已由
            # apply_trade 按成交价 + 实际费用扣除。原实现在此处再做
            # `cash += 预估成本 − 实际成本`，等于把「预估费 2.2bp − 实际佣金 0.5bp」
            # 凭空入账（实测 R36 的 923 笔买入累计 1.46 万元，占该轮收益 1.2%）。
            self._unfreeze_for_order(order)

        # Order.execute 为订单领域方法（写入成交状态），与 SQL 无关
        # pi-lens-ignore: python-sql-injection
        order.execute(abs_qty)
        return True

    def _apply_participation_limit(self, bar: Bar, order: Order, abs_qty: int) -> int:
        """按当日成交额限制单笔成交股数；流动性未知或限制关闭时原值返回。"""
        allowed = max_participating_quantity(
            bar.close, self._liquid_quote(bar), self.participation_limit
        )
        if allowed is None:
            return abs_qty
        return min(abs_qty, allowed)

    @staticmethod
    def _liquid_quote(bar: Bar) -> Decimal:
        """当日成交额（元）：缺失时以 成交量 × 收盘价 兜底。"""
        return daily_turnover(bar.amount, bar.volume, bar.close)

    @staticmethod
    def _limit_data_sane(base_price: Decimal, pre_close: Decimal) -> bool:
        """基准价与 pre_close 是否自洽（偏差不超出涨跌停带）。

        日线数据在除权/复牌等场景下会出现 pre_close 与当日 OHLC 不一致
        （实测如 pre_close=1.7 而 close=0.52）。此时若仍按 pre_close 裁剪，
        会把成交价强行顶到涨/跌停价，产生巨额「幽灵滑点」（单笔占比高达 66%）。
        判定为不自洽时放弃裁剪，改用当日实际价格。
        """
        deviation = abs(base_price - pre_close) / pre_close
        return deviation <= MAX_LIMIT_DEVIATION

    def _get_exec_price(self, bar: Bar, is_buy: bool, quantity: int = 0) -> Decimal:
        base_price = bar.open if self.match_mode == MatchMode.NEXT_OPEN else bar.close
        self._exec_price_calls += 1
        # 滑点模型在 quantity=1 时返回的是「价格调整量」（bps 模型为 price*bps，
        # tick 模型为 tick_size），需按买卖方向加到基准价上。
        # 原实现直接把返回值当成成交价，导致零滑点（返回 0）时成交价被裁剪至
        # 跌停价（买单）/ 涨停价（卖单），全部交易以错误价格成交。
        adjustment = self.slippage.calculate(base_price, 1, is_buy)
        # 冲击让价依赖本次成交量（平方根模型），与滑点同向叠加
        impact = self.impact.price_impact(base_price, quantity, self._liquid_quote(bar))
        if is_buy:
            exec_price = base_price + adjustment + impact
        else:
            exec_price = base_price - adjustment - impact

        if bar.pre_close <= Decimal("0"):
            return exec_price

        bounds = self._clip_bounds(bar, base_price)
        if bounds is None:
            return exec_price

        limit_down, limit_up = bounds
        clipped_price = max(min(exec_price, limit_up), limit_down)
        if clipped_price != exec_price:
            self._clip_count += 1
            self._clip_notional += abs(clipped_price - exec_price) * Decimal(quantity)
            logger.debug(
                f"Price Clipped: {bar.symbol} {exec_price} -> {clipped_price} "
                f"(limit [{limit_down}, {limit_up}])"
            )
        return clipped_price

    @staticmethod
    def _annotated_limit_rate(bar: Bar) -> Decimal | None:
        """数据湖标注的当日涨跌停比例；未标注（None/NaN）返回 None。

        `0` 是有效值——当日无涨跌幅限制（上市首日、注册制前 5 日、退市整理期首日等）
        ——必须与「没有标注」区分：前者不裁剪，后者回退到板块推得比例。
        """
        rate = bar.limit_rate
        if rate is None or not rate.is_finite():
            return None
        return rate

    def _clip_bounds(
        self, bar: Bar, base_price: Decimal
    ) -> tuple[Decimal, Decimal] | None:
        """当日裁剪区间（跌停价, 涨停价）；返回 None 表示不裁剪。

        比例优先级：数据湖标注 > 板块推得比例。以下情况不裁剪：
        - 标注为无涨跌幅限制（比例 0，交易日制度事实）；
        - 无标注且基准价与 pre_close 不自洽（除权/复牌数据，裁剪会产生幽灵滑点）。
        """
        annotated = self._annotated_limit_rate(bar)
        if annotated is not None:
            if annotated <= Decimal("0"):
                self._no_limit_calls += 1
                return None
            rate = annotated
        elif self._limit_data_sane(base_price, bar.pre_close):
            rate = self._get_limit_rate(bar)
        else:
            return None
        return (
            bar.pre_close * (Decimal("1") - rate),
            bar.pre_close * (Decimal("1") + rate),
        )

    def _exec_sanity_metadata(self) -> dict[str, float | int]:
        """成交价自检元数据：裁剪发生率 = 裁剪次数 / 成交价计算次数。"""
        calls = self._exec_price_calls
        return {
            "exec_price_calls": calls,
            "clip_count": self._clip_count,
            "clip_ratio": self._clip_count / calls if calls else 0.0,
            "clip_notional": _as_float(self._clip_notional),
            "no_limit_calls": self._no_limit_calls,
        }

    def _get_limit_rate(self, bar: Bar) -> Decimal:
        """涨跌停比例：按板块判定。

        符号形如 `SZ.300912`（带交易所前缀），必须先剥前缀再判代码；
        早先直接 `symbol.startswith("300")` 永远不成立，导致创业板/科创板被
        按 10% 处理，超过 +10% 的正常成交价被裁剪到 +10%，产生幽灵滑点。
        """
        from app.data.store import symbol_to_exchange_code
        from app.engine.backtest.scenarios.limit_up_down import LimitRate

        _, code = symbol_to_exchange_code(bar.symbol)
        if code.startswith(("688", "689")):
            return LimitRate.KCB
        if code.startswith(("300", "301")):
            return LimitRate.CHINEXT
        if code.startswith(("4", "8")):
            return LimitRate.BSE
        if bar.is_st:
            return LimitRate.ST
        return LimitRate.NORMAL

    def _board_flags(self, bar: Bar) -> BoardFlags:
        """涨跌停守卫所需标记（板块优先级由 LimitRate.get_rate 保证）。

        带数据湖标注时透传 `limit_rate`，使守卫按当日真实制度判定：
        无涨跌幅限制日（比例 0）不拦单也不裁剪。
        """
        from app.data.store import symbol_to_exchange_code

        _, code = symbol_to_exchange_code(bar.symbol)
        return {
            "is_st": bar.is_st,
            "is_kcb": code.startswith(("688", "689")),
            "is_chinext": code.startswith(("300", "301")),
            "limit_rate": self._annotated_limit_rate(bar),
        }

    @staticmethod
    def _share_adjustment_ratio(position: Position, bar: Bar) -> float | None:
        """持仓相对上次同步的复权比例；无需调整时返回 None。

        主判据是 ``adj_factor``：湖里因子只承载分红与拆分（不含价格涨跌），
        只在真实公司行为日跳变，因此判定阈值可以取到 ADJ_FACTOR_EPSILON，
        且是累计值 —— 持仓标的在除权日停牌时，可在复牌日一次性补齐。
        """
        factor = _as_decimal(bar.adj_factor)
        base = position.last_adj_factor
        if base > 0 and factor > 0:
            ratio = _as_float(factor / base)
            if abs(ratio - 1.0) >= ADJ_FACTOR_EPSILON:
                return ratio
        # 回退：因子不可用（缺失/恒为 1.0）时用「昨收 / 参考价」，容差挡报价噪声
        prev_close = _as_float(position.last_price)
        pre_close = _as_float(bar.pre_close)
        if prev_close > 0 and pre_close > 0:
            ratio = prev_close / pre_close
            if abs(ratio - 1.0) >= SHARE_ADJUSTMENT_TOLERANCE:
                return ratio
        return None

    def _sync_entry_state(self, symbol: str, bar: Bar) -> None:
        """建仓/加仓后同步复权基准并按收盘价标记。

        必须在建仓当日就记下因子基准：若等到次日再记，会把「建仓当日收盘后
        发生的除权」并入基准，永久漏掉该次调整。按收盘价重新标记则保证
        ``last_price`` 始终是上一根 K 线的收盘价，参考价回退判据不会把
        成交价与收盘价的偏离误判成除权。
        """
        position = self.portfolio.positions.get(symbol)
        if position is None:
            return
        position.last_adj_factor = _as_decimal(bar.adj_factor)
        position.last_price = bar.close

    def _register_default_processors(self) -> None:
        """注册复权所需的处理器。

        ``CorporateActionProcessor`` 默认注册表为空（需调用方自行 register），
        未注册时 ``process()`` 只返回失败结果，静默跳过、不改数量。
        """
        if self.action_processor is None:
            return
        from app.rule.corporate.actions import ActionType
        from app.rule.corporate.dividend import SplitProcessor

        self.action_processor.register(ActionType.SPLIT, SplitProcessor())

    def _apply_share_adjustments(
        self, ts: datetime, current_bars: dict[str, Bar]
    ) -> int:
        """按复权因子变化复权持仓数量，返回调整的持仓数。

        ETF 份额拆分（实测 159582 于 2026-07-12 做 5:1 拆分：因子 ×5、原始价
        4.991→0.929）与股票送股/转增、现金分红，都表现为复权因子跳变。
        若只按原始价标记，持仓市值会凭空消失（拆分）或凭空出现（合并），
        卖出时更会按错误数量成交。

        处理方式：数量按 ``今日因子 / 上次同步因子`` 比例放大，总成本不变；
        现金分红走同一路径（等价于分红再投资，无税/费差异时与入账等价）。
        """
        if self.action_processor is None:
            return 0
        from decimal import Decimal as _Decimal

        from app.rule.corporate.actions import ActionType, SplitAction

        adjusted = 0
        for symbol, bar in current_bars.items():
            position = self.portfolio.positions.get(symbol)
            if position is None or position.quantity <= 0:
                continue
            factor = _as_decimal(bar.adj_factor)
            ratio = self._share_adjustment_ratio(position, bar)
            if ratio is None:
                position.last_adj_factor = factor
                continue
            action = SplitAction(
                symbol=symbol,
                ex_date=ts,
                pub_date=ts,
                event_type=ActionType.SPLIT,
                split_ratio=_Decimal(str(round(ratio, 6))),
            )
            result = self.action_processor.process(
                action=action,
                quantity=position.quantity,
                cost_basis=position.cost_basis,
                available_cash=self.portfolio.available_cash,
            )
            if result.success:
                self.portfolio.apply_action_result(result)
                position.last_adj_factor = factor
                adjusted += 1
            else:
                logger.warning(
                    "Share adjustment failed: %s ratio=%.6f quantity=%s",
                    symbol,
                    ratio,
                    position.quantity,
                )
        return adjusted

    def _process_corporate_actions(
        self,
        ts: datetime,
        current_bars: dict[str, Bar],
    ) -> None:
        if self.action_processor is None:
            return

        for symbol, pos in list(self.portfolio.positions.items()):
            pending = self._pending_actions.get(symbol, [])
            if not pending:
                continue

            to_apply = [a for a in pending if a.ex_date.date() == ts.date()]
            if not to_apply:
                continue

            for action in to_apply:
                result = self.action_processor.process(
                    action=action,
                    quantity=pos.quantity,
                    cost_basis=pos.cost_basis,
                    available_cash=self.portfolio.available_cash,
                )
                if result.success:
                    self.portfolio.apply_action_result(result)
                    logger.info(
                        f"[{ts.date()}] Corporate action: {symbol} - "
                        f"{result.description}"
                    )
                else:
                    logger.warning(
                        f"[{ts.date()}] Corporate action failed: {symbol} - "
                        f"{action.description}: {result.message}"
                    )

            self._pending_actions[symbol] = [a for a in pending if a not in to_apply]

    def add_corporate_action(self, action: CorporateAction) -> None:
        from app.rule.corporate.actions import CorporateActionProcessor

        if self.action_processor is None:
            self.action_processor = CorporateActionProcessor()
        if action.symbol not in self._pending_actions:
            self._pending_actions[action.symbol] = []
        self._pending_actions[action.symbol].append(action)

    def _liquidate_all(
        self,
        all_data: dict[str, list[Bar]],
        indices: dict[str, int],
        final_ts: datetime,
    ) -> None:
        for symbol, pos in list(self.portfolio.positions.items()):
            if pos.quantity <= 0:
                continue
            bar = None
            idx = indices.get(symbol, 0) - 1
            if idx >= 0 and idx < len(all_data[symbol]):
                bar = all_data[symbol][idx]
            if bar is None:
                continue
            sell_order = Order(
                order_id=f"LIQ-{final_ts.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
                symbol=symbol,
                qty=-pos.quantity,
                created_ts=final_ts,
                status=OrderStatus.PENDING,
            )
            self.order_log.append(sell_order)
            self._execute_single_order(sell_order, bar)
        for order in list(self._pending_orders):
            if order.status == OrderStatus.PENDING:
                order.cancel("backtest_end")
                self._unfreeze_for_order(order)
        self._pending_orders.clear()

    def _apply_executor_filter(
        self,
        target_positions: dict[str, int],
        current_bars: dict[str, Bar],
    ) -> dict[str, int]:
        if self.executor is None:
            return target_positions

        config = self.executor.config
        prices = {s: bar.close for s, bar in current_bars.items()}
        current_holdings = {s: p.quantity for s, p in self.portfolio.positions.items()}

        result: dict[str, int] = {}

        for symbol, target_qty in target_positions.items():
            current_qty = current_holdings.get(symbol, 0)

            if target_qty == current_qty:
                result[symbol] = target_qty
                continue

            price = prices.get(symbol, Decimal("0"))
            if price <= Decimal("0"):
                continue

            change_qty = abs(target_qty - current_qty)
            if change_qty <= 0:
                continue

            is_sell = target_qty < current_qty
            cost = self.commission.calculate(price, change_qty, is_buy=not is_sell)
            total_cost = cost.total

            change_value = Decimal(change_qty) * price
            friction_ratio = (
                total_cost / change_value
                if change_value > Decimal("0")
                else Decimal("1")
            )

            if friction_ratio <= config.friction_threshold:
                result[symbol] = target_qty
            else:
                result[symbol] = current_qty
                logger.debug(
                    f"Friction Filter: {symbol} {current_qty} -> {target_qty} "
                    f"(friction {friction_ratio:.4f} > threshold "
                    f"{config.friction_threshold})"
                )

        for symbol, qty in current_holdings.items():
            if symbol not in result and qty > 0:
                result[symbol] = qty

        return result
