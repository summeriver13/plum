"""Obsidian 笔记渲染：把回测 report/summary 渲染成 Markdown。

供 ``export_backtest_to_obsidian.py`` 调用；也可单独复用于其它导出场景。
约定：frontmatter/表格中的收益率、回撤、alpha 为百分数值，turnover 为倍数。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')
GENERATOR = "scripts/export_backtest_to_obsidian.py"


@dataclass(frozen=True)
class BatchSpec:
    """一个回测批次的元信息。"""

    key: str  # 输出目录名与笔记前缀
    results_dir: str  # data/results 下的批次目录名
    label: str  # 展示名
    date: str
    tag: str  # 文件名/标签用的短前缀（v8 / v9）
    exposure: str  # 仓位口径说明
    data_note: str  # 数据口径说明
    max_positions_desc: str = "{select_num}（= 选股数）"  # 最大持仓描述模板
    related: tuple[str, ...] = ()  # 相关手写笔记（wikilink 目标）
    family: str = "multifactor_rounds"  # 轮次语义相同的批次族（仅族内做逐轮对照）


def to_float(value: Any) -> float | None:
    """把 '172.08%' / '30.1x' / 0.71 归一化为 float。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    try:
        return round(float(value.strip().rstrip("%xX")), 6)
    except (TypeError, ValueError):
        return None


def to_int(value: Any, default: int = -1) -> int:
    """容错整数转换（汇总 JSON 字段缺失/非法时回退默认值）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def bench_label(report: dict[str, Any]) -> str:
    """基准显示名（如 STYLE.MICROCAP → 微盘等权风格指数）；解析失败回退原始代码。"""
    symbol = str((report.get("基准对比") or {}).get("benchmark_symbol") or "")
    if not symbol:
        return "基准"
    try:
        from app.engine.backtest.analysis.benchmarks import benchmark_label

        return benchmark_label(symbol)
    except Exception:  # noqa: BLE001 — 名称只影响展示
        return symbol


def slugify(title: str) -> str:
    """标题转文件名安全片段。"""
    return INVALID_FILENAME_CHARS.sub("-", title).replace(" ", "").strip("-")


def fmt_pct(value: float | None) -> str:
    """百分数值（如 16.1）→ '+16.10%'。"""
    return "—" if value is None else f"{value:+.2f}%"


def fmt_plain(value: float | None, digits: int = 2, suffix: str = "") -> str:
    """普通数值 → 固定小数位文本。"""
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def load_summary(results_root: Path, results_dir: str) -> list[dict[str, Any]]:
    """合并批次目录下所有 rounds_summary_*.json（按轮次去重，新文件覆盖旧文件）。

    分批跑回测时每批只写自己那几轮，因此需要跨文件合并；同轮次出现多次
    （重跑、失败后重试）时以文件名时间戳最新的为准。
    """
    batch_dir = results_root / results_dir
    summaries = sorted(batch_dir.glob("rounds_summary_*.json"))
    if not summaries:
        raise FileNotFoundError(f"{batch_dir} 下没有 rounds_summary_*.json")
    merged: dict[int, dict[str, Any]] = {}
    for path in summaries:
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"汇总文件不可读或非合法 JSON: {path}（{exc}）") from exc
        for item in items:
            merged[to_int(item.get("round_id"))] = item
    ordered = [merged[key] for key in sorted(merged)]
    # 失败的轮次（只有 error 字段）不生成页面
    return [item for item in ordered if not item.get("error")]


def copy_charts(run_dir: Path, assets_dir: Path, prefix: str) -> list[tuple[str, str]]:
    """复制 charts/*/chart.png 到 assets/，返回 (图名, 笔记相对链接)。"""
    assets_dir.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str]] = []
    for png in sorted((run_dir / "charts").glob("*/chart.png")):
        name = f"{prefix}_{png.parent.name}.png"
        shutil.copyfile(png, assets_dir / name)
        items.append((png.parent.name, f"assets/{name}"))
    return items


def key_metrics(report: dict[str, Any]) -> dict[str, float | int | None]:
    """抽取跨页面复用的关键指标（百分数为百分数值）。"""
    income, risk = report.get("收益", {}), report.get("风险", {})
    adj, trade = report.get("风险调整", {}), report.get("交易行为", {})
    bench, friction = report.get("基准对比", {}), report.get("摩擦成本", {})
    return {
        "cum_return": to_float(income.get("total_return")),
        "annual_return": to_float(income.get("annual_return")),
        "sharpe": to_float(adj.get("sharpe")),
        "calmar": to_float(adj.get("calmar")),
        "max_drawdown": to_float(risk.get("max_drawdown")),
        "annual_volatility": to_float(risk.get("annual_volatility")),
        "turnover": to_float(trade.get("annual_turnover")),
        "trade_count": trade.get("total_trades"),
        "win_rate": to_float(trade.get("win_rate")),
        "avg_holding_days": to_float(trade.get("avg_holding_days")),
        "profit_factor": to_float(trade.get("profit_factor")),
        "alpha": to_float(bench.get("alpha")),
        "beta": to_float(bench.get("beta")),
        "information_ratio": to_float(bench.get("information_ratio")),
        "friction_ratio": to_float(friction.get("friction_ratio")),
    }


def front_matter(
    batch: BatchSpec, item: dict[str, Any], report: dict[str, Any]
) -> list[str]:
    """构造 YAML frontmatter。"""
    metrics = key_metrics(report)
    pairs: list[tuple[str, Any]] = [
        ("type", "回测轮次"),
        ("batch", batch.key),
        ("batch_label", batch.label),
        ("round", to_int(item.get("round_id"))),
        ("title", item["title"]),
        ("period", report.get("period")),
        ("trading_days", report.get("trading_days")),
        ("tags", ["回测", "多因子小市值", batch.tag]),
        *metrics.items(),
        ("elapsed_sec", item.get("elapsed_sec")),
        ("generator", GENERATOR),
        ("generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    lines = ["---"]
    for key, value in pairs:
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {entry}" for entry in value)
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return lines


def round_note(
    batch: BatchSpec,
    item: dict[str, Any],
    report: dict[str, Any],
    config: dict[str, str | int],
    charts: list[tuple[str, str]],
    names: dict[int, str],
) -> str:
    """渲染单轮笔记正文（frontmatter 由调用方拼接）。"""
    round_id = to_int(item.get("round_id"))
    m = key_metrics(report)
    bench = report.get("基准对比", {})
    nav = [f"[[{batch.key}/0_批次索引|← {batch.label}]]"]
    if round_id - 1 in names:
        nav.append(f"上一轮 [[{names[round_id - 1][:-3]}]]")
    if round_id + 1 in names:
        nav.append(f"下一轮 [[{names[round_id + 1][:-3]}]]")
    nav.append(f"完整报告 [[{batch.tag}-R{round_id}-完整报告]]")
    lines = [
        f"# R{round_id} {item['title']}",
        "",
        " ｜ ".join(nav),
        "",
        "> [!summary] 核心读数",
        f"> - 累计 **{fmt_pct(m['cum_return'])}**｜年化 **{fmt_pct(m['annual_return'])}**"
        f"｜Sharpe **{fmt_plain(m['sharpe'])}**｜最大回撤 **{fmt_plain(m['max_drawdown'])}%**"
        f"｜Calmar **{fmt_plain(m['calmar'])}**"
        f"｜年化波动 **{fmt_plain(m['annual_volatility'])}%**",
        f"> - 基准 {bench_label(report)} {fmt_pct(to_float(bench.get('benchmark_return')))}"
        f"（年化 {fmt_pct(to_float(bench.get('benchmark_annual')))}）"
        f"｜Alpha **{fmt_pct(m['alpha'])}**｜Beta **{fmt_plain(m['beta'])}**"
        f"｜信息比率 **{fmt_plain(m['information_ratio'])}**",
        f"> - 交易 {m['trade_count'] or '—'} 笔｜胜率 {fmt_plain(m['win_rate'])}%"
        f"｜平均持有 {fmt_plain(m['avg_holding_days'], 1)} 天"
        f"｜年换手 **{fmt_plain(m['turnover'], 1, 'x')}**"
        f"｜盈亏比 {fmt_plain(m['profit_factor'])}"
        f"｜摩擦/初始资金 {fmt_plain(m['friction_ratio'])}%",
        "",
        "## 假设",
        "",
        str(item.get("hypothesis", "")),
        "",
        "## 配置",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 选股数 select_num | {config.get('select_num', '—')} |",
        f"| 最大持仓 max_positions | {batch.max_positions_desc.format(select_num=config.get('select_num', '—'))} |",
        f"| 调仓周期 | {config.get('hold_period', '—')} |",
        f"| 仓位口径 | {config.get('仓位', batch.exposure)} |",
        f"| 因子 | {config.get('factors', '—')} |",
        f"| 过滤 | {config.get('filters', '—')} |",
        "",
        "## 图表",
        "",
    ]
    for chart_name, link in charts:
        lines.extend([f"### {chart_name}", "", f"![]({link})", ""])
    return "\n".join(lines)


def batch_index(batch: BatchSpec, rows: list[dict[str, Any]]) -> str:
    """渲染批次索引页。"""
    period = rows[0]["report"].get("period", "—") if rows else "—"
    lines = [
        "---",
        "type: 回测批次",
        f"batch: {batch.key}",
        f"label: {batch.label}",
        f"date: {batch.date}",
        "tags:",
        "  - 回测批次",
        "  - 多因子小市值",
        f"  - {batch.tag}",
        "---",
        "",
        f"# {batch.label} — 五轮回测索引",
        "",
        f"> 回测区间 **{period}**｜初始资金 100 万｜基准 中证1000（IDX.000852）",
        f"> 结果目录 `data/results/{batch.results_dir}/`",
        "",
        "## 数据与仓位口径",
        "",
        f"- 数据：{batch.data_note}",
        f"- 仓位：{batch.exposure}",
        "",
        "## 五轮对比",
        "",
        "| 轮次 | 方案 | 累计 | 年化 | Sharpe | 回撤 | Calmar | 换手 | Alpha | IR | 笔记 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        item = row["item"]
        m = key_metrics(row["report"])
        lines.append(
            f"| R{item['round_id']} | {item['title']} | {fmt_pct(m['cum_return'])} "
            f"| {fmt_pct(m['annual_return'])} | {fmt_plain(m['sharpe'])} "
            f"| {fmt_plain(m['max_drawdown'])}% | {fmt_plain(m['calmar'])} "
            f"| {fmt_plain(m['turnover'], 1, 'x')} "
            f"| {fmt_pct(m['alpha'])} | {fmt_plain(m['information_ratio'])} "
            f"| [[{row['note_name'][:-3]}]] |"
        )
    lines.extend(
        [
            "",
            "## 相关文档",
            "",
            *(f"- [[{target}]]" for target in batch.related),
            "- [[0_回测总览|← 返回回测总览]]",
            "",
            "## 复现",
            "",
            "```bash",
            "PLUM_DISABLE_QMT=1 uv run python scripts/backtest_multifactor_rounds.py"
            f" --out-dir data/results/{batch.results_dir}",
            f"uv run python {GENERATOR} --batches {batch.key}",
            "```",
        ]
    )
    return "\n".join(lines)
