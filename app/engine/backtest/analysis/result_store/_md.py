"""Markdown 报告生成 — 四章节结构，图表随文走。"""

from __future__ import annotations

import json
from pathlib import Path

from app.config.paths import STORAGE_META_FILE


def _num(value: object, default: float = 0.0) -> float:
    """容错转 float（numpy 标量/NULL 均可）。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    """容错转 int（NULL/非数值字符串返回 default）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _write_lines(path: Path, lines: list[str]) -> None:
    """写入文本行（磁盘错误转 RuntimeError，保留原始原因）。"""
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"报告写入失败: {path}（{exc}）") from exc


# ── 图表导读 ──────────────────────────────────────────

CHART_NOTES: dict[str, str] = {
    "01_净值曲线": (
        "上图是策略净值曲线（起始=1.00），下图是回撤深度（负数）。"
        "净值长期在 1.00 以上=正收益。回撤面积越小越好，回到零轴越快越好。"
    ),
    "02_月度收益热力图": (
        "横轴月/纵轴年，颜色=收益率。绿色=盈利、红色=亏损。"
        "关注连续红色月份——超过 3 个月连亏说明策略在某段时间失效。"
    ),
    "03_年度收益": (
        "每年一根柱子，绿色=赚钱/红色=亏钱。"
        "如果连续两年以上是红色，策略可能有根本性问题。"
    ),
    "04_日收益分布": (
        "蓝柱=实际日收益分布，虚线=正态分布。"
        "如果左边尾巴比正态肥（峰度>1），说明极端亏损比预期多。偏度<0=亏的日子比赚的日子更极端。"
    ),
    "05_滚动夏普回撤": (
        "上图夏普>0 说明过去一年赚钱，下图是过去一年的最大回撤。"
        "如果夏普长期低于 0 或回撤越来越大，策略正在恶化。"
    ),
    "06_交易散点图": (
        "每笔交易一个点：横轴持仓天数，纵轴盈亏损益，圆圈大小=交易数量。"
        "左上角（持仓短+大赚）=短线获利能力强；右下角（持仓长+大亏）=止损有问题。"
    ),
    "07_QQ图": (
        "蓝点应落在红色虚线上才符合正态分布。"
        "两端偏离红线=厚尾（大亏大赚比预期多）。整体弯曲=偏态。"
    ),
    "08_水下热力图": (
        "颜色越深（红）回撤越大，白色=无回撤。"
        "如果每年同一月份出现深色块，策略有季节性弱点。"
    ),
    "09_滚动年化收益": (
        "这条线越平稳越好。如果后半段明显下滑，说明策略在过去一年失效了。"
        "虚线是均值，低于均值的时间越长越危险。"
    ),
    "10_回撤特写": (
        "每条线是独立的一次回撤，从回撤前高点=1.00 开始。"
        "线越早回到 1.0 以上说明恢复越快。如果多条线长期低于 1.0，策略有生存危机。"
    ),
    "11_周历效应": (
        "箱体中间线是中位数，箱子上下是 25%/75% 分位。"
        "如果某几天中位数明显偏离零，说明存在周历效应——策略在这些日子系统性跑赢或跑输。"
    ),
    "12_仓位暴露": (
        "上图是每天的总仓位占比，下图是仓位分布直方图。"
        "高仓位+高收益=策略择时能力强。高仓位+低收益=可能是在赌博。空仓天数多说明策略有选择性。"
    ),
}


# ── 图表分组 ──────────────────────────────────────────

# 图表按报告章节分组，图随文走
CHART_SECTIONS: dict[str, list[str]] = {
    "业绩": ["01_净值曲线", "02_月度收益热力图", "03_年度收益"],
    "风险": ["05_滚动夏普回撤", "08_水下热力图", "09_滚动年化收益"],
    "交易": ["06_交易散点图", "10_回撤特写", "12_仓位暴露"],
    "验证": ["04_日收益分布", "07_QQ图", "11_周历效应"],
}


# ── 股票名称映射 ──────────────────────────────────────


def _load_stock_names() -> dict[str, str]:
    """从本地 catalog 加载股票代码→名称映射。"""
    meta_path = STORAGE_META_FILE
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("stock_names", {})
    except Exception:
        return {}


def _resolve_index_name(symbol: str) -> str:
    """将基准代码解析为可读名称（如 IDX.000852 → 中证1000、STYLE.MICROCAP → 微盘等权风格指数）。"""
    if not symbol:
        return ""
    try:
        from app.engine.backtest.analysis.benchmarks import benchmark_label

        return benchmark_label(symbol)
    except Exception:
        return ""


def _build_strategy_stats_table(stats: dict) -> list[str]:
    """构建多策略归因表格。"""
    lines: list[str] = []
    lines.append("#### 策略归因")
    lines.append("")
    lines.append("| 策略 | 权重 | 买入信号 | 卖出信号 | 去重股票数 |")
    lines.append("|------|------|----------|----------|------------|")
    for sid, st in sorted(stats.items()):
        w = st.get("weight", 0)
        lines.append(
            f"| {sid} | {w:.1%} | {st.get('buy', 0)} | "
            f"{st.get('sell', 0)} | {st.get('symbol_count', 0)} |"
        )
    lines.append("")
    return lines


# ── MD 报告生成 ───────────────────────────────────────


def _write_md_report(run_dir: Path, report, meta: dict | None = None) -> None:
    """生成 Markdown 回测报告 — 四章节结构（业绩/风险/交易/验证）。

    Args:
        run_dir: 结果目录路径
        report: ComprehensiveReport 对象（可为 None）
        meta: 元数据字典
    """
    from app.engine.backtest.analysis.charts import CHART_SPECS

    lines: list[str] = []
    sid = (meta or {}).get("strategy_id", "")

    # ── Header ──
    lines.append(f"# 回测报告: {sid}" if sid else "# 回测报告")
    lines.append("")

    has_report = report is not None and hasattr(report, "to_dict")
    d: dict = report.to_dict() if has_report else {}

    # 周期行
    period = d.get("period", "")
    td = d.get("trading_days", "")
    if period:
        lines.append(f"**{period}** | {td} 交易日")
        lines.append("")

    # 概览摘要
    overview = []
    if has_report:
        ret = d["收益"]["total_return"]
        sharpe = d["风险调整"]["sharpe"]
        mdd = d["风险"]["max_drawdown"]
        overview.append(f"累计收益 **{ret}**")
        overview.append(f"Sharpe **{sharpe}**")
        overview.append(f"最大回撤 **{mdd}**")
        bm = d.get("基准对比")
        if bm and bm.get("alpha", "0.00%") not in ("0.00%", "0.0000%"):
            overview.append(f"Alpha **{bm['alpha']}** (β={bm['beta']})")
    if overview:
        lines.append("> " + " | ".join(overview))
        lines.append("")

    if not has_report:
        lines.append("")
        _write_lines(run_dir / "report.md", lines)
        return

    # 标的列表
    strategy_raw = (meta or {}).get("strategy", "")
    if strategy_raw:
        names = _load_stock_names()
        codes = [c.strip() for c in strategy_raw.split(",") if c.strip()]
        if codes:
            lines.append("### 标的列表")
            lines.append("")
            lines.append("| 代码 | 名称 |")
            lines.append("|------|------|")
            for code in codes:
                name = names.get(code.lower(), "-")
                lines.append(f"| {code} | {name} |")
            lines.append("")
        lines.append("")

    charts_dir = run_dir / "charts"
    chart_spec_map = dict(CHART_SPECS)

    # ═══════════════════════════════════════════════════
    # Section 1: 业绩
    # ═══════════════════════════════════════════════════
    lines.append("## 1. 业绩")
    lines.append("")

    _md_table(
        lines,
        "收益指标",
        [
            ("累计收益", d["收益"]["total_return"]),
            ("年化收益", d["收益"]["annual_return"]),
            ("日胜率", d["收益"]["win_ratio"]),
            ("均盈", d["收益"]["avg_win"]),
            ("均亏", d["收益"]["avg_loss"]),
        ],
    )

    # 基准对比
    bm = d.get("基准对比")
    if bm and bm.get("alpha", "0.00%") not in ("0.00%", "0.0000%"):
        bm_symbol = bm.get("benchmark_symbol", "")
        bm_label = _resolve_index_name(bm_symbol)
        heading = (
            f"基准对比 — {bm_label}（{bm_symbol}）"
            if bm_label and bm_label != bm_symbol
            else f"基准对比 — {bm_symbol}"
        )
        _md_table(
            lines,
            heading,
            [
                ("基准总收益", bm["benchmark_return"]),
                ("基准年化", bm["benchmark_annual"]),
                ("Alpha", bm["alpha"]),
                ("Beta", str(bm["beta"])),
                ("跟踪误差", bm["tracking_error"]),
                ("信息比率", str(bm["information_ratio"])),
            ],
        )

    # 闲置资金收益（停泊）：收益来源拆解
    ci = d.get("闲置资金收益")
    if ci:
        lines.append("#### 闲置资金收益（国债逆回购停泊）")
        lines.append("")
        lines.append("| 口径 | 年化 | Sharpe | 回撤 | Calmar |")
        lines.append("|------|------|--------|------|--------|")
        lines.append(
            f"| 含停泊 | {d['收益']['annual_return']} | {d['风险调整']['sharpe']} "
            f"| {d['风险']['max_drawdown']} | {d['风险调整']['calmar']} |"
        )
        lines.append(
            f"| 剥离停泊（近似） | {ci['annual_return_ex']} | {ci['sharpe_ex']} "
            f"| {ci['max_drawdown_ex']} | {ci['calmar_ex']} |"
        )
        lines.append("")
        _md_table(
            lines,
            "停泊明细",
            [
                ("累计利息", f"{ci['cash_interest_total']:,.2f} 元"),
                ("平均闲置资金占比", ci["cash_ratio"]),
                (
                    "停泊年化率",
                    f"{ci['cash_yield_rate']}（停泊比例 {ci['park_ratio']}）",
                ),
                ("对年化贡献（算术）", f"{report.cash_interest_annual * 100:+.2f} pp"),
                ("占累计收益金额", ci["cash_interest_share"]),
            ],
        )
        lines.append(
            "> 算术贡献按 Σ(当日利息 ÷ 前一日净值) 年化，与日收益口径可加；"
            "实际 CAGR 差额会高于此值，因为利息再投资会放大仓位规模。"
        )
        lines.append("> 剥离对照仅减去利息本金、不重算仓位，因此是**下界**口径。")
        lines.append("")

    # 策略归因（仅多策略回测）
    strategy_stats = (meta or {}).get("strategy_stats", {})
    if strategy_stats and len(strategy_stats) > 1:
        stat_lines = _build_strategy_stats_table(strategy_stats)
        lines.extend(stat_lines)

    # 年度收益
    yr = d.get("yearly_returns", {})
    if yr and len(yr) > 1:
        _md_table(lines, "年度收益", [(str(y), f"{r:+.2%}") for y, r in yr.items()])

    # 月度收益（最佳/最差并列）
    mr = d.get("monthly_returns", {})
    if mr and len(mr) > 1:
        best = sorted(mr.items(), key=lambda x: x[1], reverse=True)[:3]
        worst = sorted(mr.items(), key=lambda x: x[1])[:3]
        lines.append("#### 月度收益")
        lines.append("")
        lines.append("| 最佳月份 | | 最差月份 | |")
        lines.append("|----------|--------|----------|--------|")
        for (bm_n, bm_v), (wm_n, wm_v) in zip(best, worst):
            lines.append(f"| {bm_n} | {bm_v:+.2%} | {wm_n} | {wm_v:+.2%} |")
        lines.append("")

    _embed_charts(lines, charts_dir, chart_spec_map, "业绩")

    # ═══════════════════════════════════════════════════
    # Section 2: 风险
    # ═══════════════════════════════════════════════════
    lines.append("## 2. 风险")
    lines.append("")

    _md_table(
        lines,
        "风险指标",
        [
            ("最大回撤", d["风险"]["max_drawdown"]),
            ("回撤持续", f"{d['风险']['max_drawdown_days']} 天"),
            ("年化波动率", d["风险"]["annual_volatility"]),
            ("下行波动率", d["风险"]["downside_volatility"]),
            ("VaR 95", d["风险"]["VaR_95"]),
            ("CVaR 95", d["风险"]["CVaR_95"]),
            ("最大单日亏损", d["风险"]["max_single_day_loss"]),
        ],
    )

    _md_table(
        lines,
        "风险调整",
        [
            ("Sharpe", str(d["风险调整"]["sharpe"])),
            ("Sortino", str(d["风险调整"]["sortino"])),
            ("Calmar", str(d["风险调整"]["calmar"])),
            ("Omega", str(d["风险调整"]["omega"])),
        ],
    )

    # 回撤区间
    drawdown_intervals = d["风险"].get("drawdown_intervals", [])
    if drawdown_intervals:
        lines.append("#### 回撤区间")
        lines.append("")
        lines.append("| 开始 | 结束 | 深度 | 持续天数 | 恢复 |")
        lines.append("|------|------|------|----------|------|")
        for dd in drawdown_intervals:
            depth = f"{dd['depth']:.1%}"
            recovery = f"{dd['recovery']}天" if dd.get("recovery", -1) > 0 else "未恢复"
            lines.append(
                f"| {dd['start']} | {dd['end']} | {depth} | {dd['days']} | {recovery} |"
            )
        lines.append("")
    else:
        dd_start = d["风险"].get("max_drawdown_start", "")
        dd_end = d["风险"].get("max_drawdown_end", "")
        if dd_start or dd_end:
            lines.append("#### 回撤区间")
            lines.append("")
            lines.append("| 开始 | 结束 |")
            lines.append("|------|------|")
            lines.append(f"| {dd_start} | {dd_end} |")
            lines.append("")

    # 日收益分布
    daily = d.get("daily_returns", [])
    if daily and len(daily) > 5:
        import numpy as np

        arr = np.array(daily, dtype=float)
        dist_rows = [
            ("最小值", f"{np.min(arr):.2%}"),
            ("P5", f"{np.percentile(arr, 5):.2%}"),
            ("P25", f"{np.percentile(arr, 25):.2%}"),
            ("中位数", f"{np.median(arr):.2%}"),
            ("P75", f"{np.percentile(arr, 75):.2%}"),
            ("P95", f"{np.percentile(arr, 95):.2%}"),
            ("最大值", f"{np.max(arr):.2%}"),
            ("正收益日占比", f"{_num(np.mean(arr > 0)):.1%}"),
        ]
        _md_table(lines, "日收益分布", dist_rows)

    _embed_charts(lines, charts_dir, chart_spec_map, "风险")

    # ═══════════════════════════════════════════════════
    # Section 3: 交易
    # ═══════════════════════════════════════════════════
    lines.append("## 3. 交易")
    lines.append("")

    _md_table(
        lines,
        "交易行为",
        [
            ("总交易次数", str(d["交易行为"]["total_trades"])),
            ("胜率", d["交易行为"]["win_rate"]),
            ("平均持仓(天)", str(d["交易行为"]["avg_holding_days"])),
            ("年化换手率", d["交易行为"]["annual_turnover"]),
            ("空仓比", d["交易行为"]["idle_ratio"]),
            ("盈亏比", str(d["交易行为"]["profit_factor"])),
        ],
    )

    # 摩擦成本
    fc = d.get("摩擦成本")
    if fc and fc.get("total_friction", 0) > 0:
        rows = [
            ("总摩擦成本", f"{fc['total_friction']:,.2f} 元"),
            ("摩擦比率", fc["friction_ratio"]),
        ]
        if fc.get("max_slippage_ratio"):
            rows.append(("最大单笔滑点/成交额", fc["max_slippage_ratio"]))
        anomaly = _int(fc.get("slippage_anomaly_count", 0) or 0)
        rows.append(
            (
                "异常滑点笔数",
                "0（正常）"
                if anomaly == 0
                else f"⚠️ {anomaly} 笔 / {fc.get('slippage_anomaly_cost', 0):,.0f} 元",
            )
        )
        _md_table(lines, "交易成本", rows)
        if anomaly:
            lines.append("")
            lines.append(
                "> ⚠️ **成交价自检未通过**：存在滑点占成交额 > 2% 的成交，"
                "正常冲击成本应在 bp 级。请核查涨跌停裁剪与 `pre_close` 数据一致性。"
            )

    # 成交价自检：裁剪发生率（静默故障的伴随指标，正常回测低位）
    sc = d.get("成交价自检")
    if sc and sc.get("exec_price_calls"):
        _md_table(
            lines,
            "成交价自检",
            [
                ("成交价计算次数", str(sc["exec_price_calls"])),
                ("涨跌停裁剪笔数", str(sc["clip_count"])),
                ("裁剪发生率", sc["clip_ratio"]),
                ("裁剪调整金额", f"{sc['clip_notional']:,.2f} 元"),
                ("无涨跌幅限制日成交", str(sc.get("no_limit_calls", 0))),
            ],
        )

    _embed_charts(lines, charts_dir, chart_spec_map, "交易")

    # ═══════════════════════════════════════════════════
    # Section 4: 验证
    # ═══════════════════════════════════════════════════
    lines.append("## 4. 验证")
    lines.append("")

    _md_table(
        lines,
        "策略真实性",
        [
            ("半期 Sharpe", str(d["真实性"]["half_sharpe"])),
            ("偏度", str(d["真实性"]["daily_skew"])),
            ("峰度", str(d["真实性"]["daily_kurtosis"])),
            ("恢复因子", str(d["真实性"]["recovery_factor"])),
        ],
    )

    _embed_charts(lines, charts_dir, chart_spec_map, "验证")

    _write_lines(run_dir / "report.md", lines)


# ── 辅助 ──────────────────────────────────────────────


def _md_table(lines: list[str], title: str, rows: list[tuple[str, str]]) -> None:
    """向 lines 追加一个指标表格。"""
    lines.append(f"#### {title}")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")


def _embed_charts(
    lines: list[str],
    charts_dir: Path,
    spec_map: dict[str, str],
    section: str,
) -> None:
    """嵌入指定章节的图表。"""
    for dirname in CHART_SECTIONS.get(section, []):
        img = charts_dir / dirname / "chart.png"
        if img.exists():
            rel = f"charts/{dirname}/chart.png"
            desc = spec_map.get(dirname, dirname)
            lines.append(f"*{dirname} — {desc}*")
            lines.append("")
            lines.append(f"![]({rel})")
            lines.append("")
            note = CHART_NOTES.get(dirname)
            if note:
                lines.append(f"> {note}")
                lines.append("")
