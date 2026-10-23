"""把回测结果导出为 Obsidian 笔记。

读取 ``data/results/<批次>/`` 下的汇总 JSON、各轮 report.json / report.md 与图表，
在 ``docs/策略与回测/回测结果/`` 生成：

- ``0_回测总览.md``：跨批次与逐轮对照
- ``<批次>/0_批次索引.md``：五轮对比表 + 链接
- ``<批次>/<标签>-R<id>-<标题>.md``：每轮一页（frontmatter 指标 + 图表）
- ``<批次>/<标签>-R<id>-完整报告.md``：引擎 report.md 副本
- ``<批次>/assets/``：图表副本（笔记内相对链接嵌入）

前置条件：批次目录下存在 ``rounds_summary_*.json``（取最新一份）。

用法::

    uv run python scripts/export_backtest_to_obsidian.py --batches v8_final v9_1 family_10d dd_v10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from obsidian_notes import (
    BatchSpec,
    batch_index,
    copy_charts,
    fmt_pct,
    fmt_plain,
    front_matter,
    key_metrics,
    load_summary,
    round_note,
    slugify,
    to_int,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "策略与回测" / "回测结果"


def load_report(run_dir: Path) -> dict[str, Any]:
    """读取单轮 report.json；缺失或损坏时返回空字典。"""
    try:
        return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


BATCHES: dict[str, BatchSpec] = {
    "v8_final": BatchSpec(
        key="v8_final",
        results_dir="multifactor_rounds_v8_final",
        label="v8 市值与退市修正版",
        date="2026-09-15",
        tag="v8",
        exposure="排名加权（strength = 全市场排名 ÷ select_num），max_positions 落到引擎默认 8",
        max_positions_desc="8（引擎默认值，select_num 未生效）",
        data_note=(
            "QMT PIT 市值、退市注册表生效、observed 涨跌停标注改为 NaN，"
            "行情副本干净重建（5,444 只，含退市股）"
        ),
        related=(
            "多因子小市值五轮回测报告_v8市值与退市修正版_20260915",
            "多因子小市值_v8结果深挖_20260915",
        ),
    ),
    "v9_1": BatchSpec(
        key="v9_1",
        results_dir="multifactor_rounds_v9_1",
        label="v9.1 满仓口径修复版",
        date="2026-09-16",
        tag="v9_1",
        exposure="等权、目标仓位 98%，并修掉 ST 分母错配与卖出回款预算两个执行缺陷（实际仓位 96.5%）",
        data_note=(
            "干净重建副本；执行层三修：QMT 补数据默认关闭（只补缺不覆盖）、"
            "选股前置 PIT ST 剔除、买入预算计入当日卖出回款"
        ),
        related=(
            "多因子小市值_回撤控制研究_v10方案_20260916",
            "多因子小市值_v8结果深挖_20260915",
        ),
    ),
    "family_10d": BatchSpec(
        key="family_10d",
        results_dir="family_v9_10d",
        label="v9 策略族 @ 10 日口径",
        date="2026-09-16",
        tag="fam10",
        exposure="等权、目标仓位 98%（信号模式，与 v9.1 同口径）",
        data_note=(
            "同 v9.1（干净副本 + 执行层三修）；唯一变量 = 调仓周期 5 日 → 10 日"
        ),
        related=("多因子小市值_回撤控制研究_v10方案_20260916",),
    ),
    "dd_v10": BatchSpec(
        key="dd_v10",
        results_dir="drawdown_rounds_v10",
        label="v10 降回撤实验",
        date="2026-09-16",
        tag="v10",
        exposure="目标数量模式（等权再平衡，支持分批减仓与回补）；各轮基础仓位与趋势降档不同",
        family="drawdown_rounds",
        data_note=(
            "干净副本 + v9.1 执行层修复；风控叠加为国证2000（IDX.399303）MA120 "
            "跌破降档，信号滞后一日"
        ),
        related=("多因子小市值_回撤控制研究_v10方案_20260916",),
    ),
    "v10_clean": BatchSpec(
        key="v10_clean",
        results_dir="_verify_clean_v10_all",
        label="v10 干净口径重跑（37 轮）",
        date="2026-09-18",
        tag="v10c",
        exposure="目标数量模式；各轮基础仓位与趋势降档与 dd_v10 完全相同，仅更换数据口径",
        family="drawdown_rounds",
        data_note=(
            "行情副本于 2026-09-16 22:51 干净重建（湖派生 + 不再被 QMT 直读市值覆盖）后的重跑；"
            "本批与 dd_v10 的差值即「污染口径」影响，dd_v10 已作废"
        ),
        related=("多因子小市值_回撤控制研究_v10方案_20260916",),
    ),
}

# 各轮配置（v8/v9 相同，来自 backtest_multifactor_rounds.py 的 ROUNDS 定义）
ROUND_CONFIG: dict[int, dict[str, str | int]] = {
    1: {
        "select_num": 20,
        "hold_period": "5D",
        "factors": "总市值（升序，最小市值优先）",
        "filters": "成交额分位 ≥ 5%（宽松）；总市值 > 0",
    },
    2: {
        "select_num": 20,
        "hold_period": "5D",
        "factors": "总市值（升序，最小市值优先）",
        "filters": "成交额分位 ≥ 20%（严格）；总市值 > 0",
    },
    3: {
        "select_num": 20,
        "hold_period": "5D",
        "factors": "总市值（升序，最小市值优先）",
        "filters": "成交额分位 ≥ 20%；总市值 > 0；市值分位 ≤ 0.9（剔除微盘 10%）",
    },
    4: {
        "select_num": 30,
        "hold_period": "20D",
        "factors": "总市值（升序，最小市值优先）",
        "filters": "成交额分位 ≥ 20%；总市值 > 0；市值分位 ≤ 0.9（剔除微盘 10%）",
    },
    5: {
        "select_num": 25,
        "hold_period": "10D",
        "factors": "总市值（升序，最小市值优先）",
        "filters": "成交额分位 ≥ 20%；总市值 > 0；市值分位 ≤ 0.9 且 ≥ 0.1（剔除两端 10%）",
    },
}


# v9 策略族在 10 日口径下的重排（family_v9_10d）：除调仓周期外配置同 ROUND_CONFIG
ROUND_CONFIG_FAMILY_10D: dict[int, dict[str, str | int]] = {
    rid: {**cfg, "hold_period": "10D"} for rid, cfg in ROUND_CONFIG.items()
}


# v10 降回撤实验各轮配置（来自 backtest_drawdown_rounds.py 的 ROUNDS 定义）
_V10_COMMON: dict[str, str | int] = {
    "select_num": 20,
    "hold_period": "5D",
    "factors": "总市值（升序）",
    "filters": "成交额分位 ≥ 20%；总市值 > 0",
}
ROUND_CONFIG_V10: dict[int, dict[str, str | int]] = {
    1: {**_V10_COMMON, "仓位": "满仓 100%（目标数量模式对照）"},
    2: {**_V10_COMMON, "仓位": "固定 60%"},
    3: {**_V10_COMMON, "仓位": "满仓，国证2000 跌破 MA120 → 半仓（平均 ~79%）"},
    4: {**_V10_COMMON, "仓位": "固定 60% × MA120 降档（走坏 → 30%）"},
    5: {
        **_V10_COMMON,
        "select_num": 40,
        "filters": "成交额分位 ≥ 20%；总市值 > 0；市值分位 ≤ 0.9（剔除微盘 10%）",
        "仓位": "固定 60% × MA120 降档；40 只分散",
    },
    6: {**_V10_COMMON, "仓位": "固定 70% × MA120 降档（走坏 → 35%）"},
    7: {**_V10_COMMON, "仓位": "固定 70% × MA120 深降档（走坏 → 24.5%）"},
    8: {**_V10_COMMON, "仓位": "固定 65% × MA120 降档（走坏 → 32.5%）"},
    9: {**_V10_COMMON, "仓位": "固定 65% × MA60 降档（更快、更易骗线）"},
    10: {**_V10_COMMON, "仓位": "固定 65% × MA90 降档"},
    11: {**_V10_COMMON, "仓位": "固定 65% × MA150 降档"},
    12: {**_V10_COMMON, "仓位": "固定 65% × MA200 降档（年线）"},
    13: {
        **_V10_COMMON,
        "仓位": "固定 65% × MA120 降档；暴露信号后移 3 个交易日（锁死压力）",
    },
    14: {
        **_V10_COMMON,
        "仓位": "固定 65% × MA120 降档；冲击系数 ×2、参与率上限 ×0.5（流动性压力）",
    },
    15: {**_V10_COMMON, "hold_period": "10D", "仓位": "满仓（调仓频率扫描）"},
    16: {**_V10_COMMON, "hold_period": "20D", "仓位": "满仓（调仓频率扫描）"},
    17: {
        **_V10_COMMON,
        "hold_period": "1D",
        "仓位": "满仓，日频等权再平衡（频率扫描上界）",
    },
    18: {**_V10_COMMON, "仓位": "固定 65% × 微盘等权 MA120 降档（走坏 → 32.5%）"},
    19: {**_V10_COMMON, "仓位": "固定 65% × 微盘等权 MA60 降档"},
    20: {**_V10_COMMON, "仓位": "固定 65% × 微盘等权 MA120 深降档（走坏 → 19.5%）"},
    21: {
        **_V10_COMMON,
        "hold_period": "10D",
        "仓位": "10 日调仓 × 国证2000 MA120（65%/32.5%）",
    },
    22: {
        **_V10_COMMON,
        "hold_period": "10D",
        "仓位": "10 日调仓 × 微盘等权 MA120（65%/32.5%）",
    },
    23: {**_V10_COMMON, "hold_period": "8D", "仓位": "满仓（频率稳健性）"},
    24: {**_V10_COMMON, "hold_period": "12D", "仓位": "满仓（频率稳健性）"},
    25: {**_V10_COMMON, "hold_period": "15D", "仓位": "满仓（频率稳健性）"},
    26: {
        **_V10_COMMON,
        "hold_period": "10D",
        "仓位": "满仓；冲击 ×2、参与率 ×0.5（频率压力测）",
    },
    27: {**_V10_COMMON, "仓位": "满仓；全成本归零（归因对照，非实盘口径）"},
    28: {
        **_V10_COMMON,
        "hold_period": "10D",
        "仓位": "满仓；全成本归零（归因对照，非实盘口径）",
    },
    29: {**_V10_COMMON, "仓位": "满仓；最小交易阈值 2%（归因对照）"},
    30: {
        **_V10_COMMON,
        "hold_period": "10D",
        "仓位": "满仓；最小交易阈值 2%（归因对照）",
    },
    31: {
        **_V10_COMMON,
        "hold_period": "10D",
        "仓位": "70% × MA120 降档（走坏 → 35%）",
    },
    32: {
        **_V10_COMMON,
        "hold_period": "10D",
        "仓位": "70% × MA120 深降档（走坏 → 24.5%）",
    },
    33: {
        **_V10_COMMON,
        "hold_period": "20D",
        "仓位": "20 日调仓 × MA120（65%/32.5%）（频率稳健性）",
    },
    34: {
        **_V10_COMMON,
        "hold_period": "10D",
        "仓位": "10 日调仓 × MA150（65%/32.5%）（均线稳健性）",
    },
    35: {
        **_V10_COMMON,
        "hold_period": "30D",
        "仓位": "满仓（摩擦边界：持有期 30 日）",
    },
    36: {
        **_V10_COMMON,
        "hold_period": "40D",
        "仓位": "满仓（摩擦边界：持有期 40 日）",
    },
    37: {
        **_V10_COMMON,
        "hold_period": "20D",
        "仓位": "20 日调仓 × 国证2000 MA120（65%/32.5%）（低摩擦 + 低 beta 叠加）",
    },
}

# 批次 → 轮次配置表（未登记的批次回退到 ROUND_CONFIG）
ROUND_CONFIGS: dict[str, dict[int, dict[str, str | int]]] = {
    "dd_v10": ROUND_CONFIG_V10,
    "v10_clean": ROUND_CONFIG_V10,  # 干净口径重跑：各轮配置与 dd_v10 完全相同
    "family_10d": ROUND_CONFIG_FAMILY_10D,
}


def note_names(spec: BatchSpec, summary: list[dict[str, Any]]) -> dict[int, str]:
    """预计算各轮笔记文件名（供上一轮/下一轮链接）。"""
    names: dict[int, str] = {}
    for item in summary:
        rid = to_int(item.get("round_id"))
        names[rid] = (
            f"{spec.tag}-R{rid}-{slugify(str(item.get('title', '')))}.md".replace(
                " ", ""
            )
        )
    return names


def export_round(
    spec: BatchSpec, item: dict[str, Any], out_dir: Path, names: dict[int, str]
) -> dict[str, Any]:
    """导出单轮：笔记 + 完整报告副本 + 图表。"""
    run_dir = Path(str(item["run_dir"]))
    report = load_report(run_dir)
    round_id = to_int(item.get("round_id"))
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = copy_charts(run_dir, out_dir / "assets", f"R{round_id}")
    note = "\n".join(front_matter(spec, item, report))
    config = ROUND_CONFIGS.get(spec.key, ROUND_CONFIG).get(round_id, {})
    note += "\n\n" + round_note(spec, item, report, config, charts, names)
    note_name = names[round_id]
    (out_dir / note_name).write_text(note, encoding="utf-8")
    (out_dir / f"{spec.tag}-R{round_id}-完整报告.md").write_text(
        f"> 引擎原始报告 · 返回 [[{note_name[:-3]}]]\n\n"
        + (run_dir / "report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return {
        "item": item,
        "report": report,
        "note_name": note_name,
        "charts": len(charts),
    }


def export_batch(spec: BatchSpec, output_dir: Path) -> dict[str, Any]:
    """导出整个批次并返回总览素材。"""
    summary = load_summary(RESULTS_ROOT, spec.results_dir)
    out_dir = output_dir / spec.key
    names = note_names(spec, summary)
    rows = [export_round(spec, item, out_dir, names) for item in summary]
    (out_dir / "0_批次索引.md").write_text(batch_index(spec, rows), encoding="utf-8")
    return {"spec": spec, "rows": rows}


def overview(entries: list[dict[str, Any]]) -> str:
    """渲染跨批次总览页（含逐轮对照）。"""
    lines = [
        "---",
        "type: 回测总览",
        "tags:",
        "  - 回测",
        "  - 多因子小市值",
        f"generated: {datetime.now():%Y-%m-%d %H:%M}",
        "---",
        "",
        "# 回测结果总览",
        "",
        "> 同一策略（截面小市值五轮）在不同数据/执行口径下的结果汇总。",
        "> frontmatter 中的收益率、回撤、alpha 为百分数值；turnover 为倍数。",
        "",
        "## 批次一览",
        "",
        "| 批次 | 日期 | 数据/执行口径 | 最佳轮次（年化） | Calmar | 索引 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        spec = entry["spec"]
        best = max(
            entry["rows"],
            key=lambda row: key_metrics(row["report"])["annual_return"] or -99,
        )
        best_metrics = key_metrics(best["report"])
        lines.append(
            f"| {spec.label} | {spec.date} | {spec.data_note}"
            f" | R{best['item']['round_id']}（{fmt_pct(best_metrics['annual_return'])}）"
            f" | {fmt_plain(best_metrics['calmar'])}"
            f" | [[{spec.key}/0_批次索引]] |"
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(entry["spec"].family, []).append(entry)
    # 取成员最多的批次族（并列时取较新的），族内做逐轮对照
    group = max(groups.values(), key=lambda items: len(items))
    if len(group) >= 2:
        left, right = group[-2], group[-1]
        lines.extend(
            [
                "",
                f"## 逐轮对照（{left['spec'].tag} → {right['spec'].tag}）",
                "",
                f"| 轮次 | 方案 | {left['spec'].tag} 年化 | {right['spec'].tag} 年化 "
                f"| {left['spec'].tag} 回撤 → {right['spec'].tag} 回撤 "
                f"| {left['spec'].tag} Calmar → {right['spec'].tag} Calmar "
                f"| {left['spec'].tag} IR → {right['spec'].tag} IR |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        right_by_id = {to_int(r["item"].get("round_id")): r for r in right["rows"]}
        for row in left["rows"]:
            rid = to_int(row["item"].get("round_id"))
            if rid not in right_by_id:
                continue
            a = key_metrics(row["report"])
            b = key_metrics(right_by_id[rid]["report"])
            lines.append(
                f"| R{rid} | {row['item']['title']} | {fmt_pct(a['annual_return'])} "
                f"| {fmt_pct(b['annual_return'])} | {fmt_plain(a['max_drawdown'])}% → "
                f"{fmt_plain(b['max_drawdown'])}% | {fmt_plain(a['calmar'])} → "
                f"{fmt_plain(b['calmar'])} | {fmt_plain(a['information_ratio'])} → "
                f"{fmt_plain(b['information_ratio'])} |"
            )
    lines.extend(
        [
            "",
            "## 导出方式",
            "",
            "```bash",
            "uv run python scripts/export_backtest_to_obsidian.py --batches "
            + " ".join(entry["spec"].key for entry in entries),
            "```",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="导出回测结果为 Obsidian 笔记")
    parser.add_argument("--batches", nargs="*", default=list(BATCHES))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for key in args.batches:
        if key not in BATCHES:
            raise SystemExit(f"未知批次: {key}（可选: {', '.join(BATCHES)}）")
        spec = BATCHES[key]
        print(f"导出 {spec.label} ...", flush=True)
        entry = export_batch(spec, output_dir)
        entries.append(entry)
        print(f"  → {output_dir / spec.key}（{len(entry['rows'])} 轮）", flush=True)
    (output_dir / "0_回测总览.md").write_text(overview(entries), encoding="utf-8")
    print(f"总览: {output_dir / '0_回测总览.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
