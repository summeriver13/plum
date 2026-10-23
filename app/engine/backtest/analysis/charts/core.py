"""图表公共样式 — 配色、字体、布局工具。"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib

    matplotlib.use("Agg")  # 非交互后端，消除 tkinter 线程警告
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.debug("matplotlib 未安装，图表功能不可用")

# ── 配色方案 ──────────────────────────────────────────

PALETTE = {
    "equity": "#1f77b4",  # 策略净值 - 蓝
    "benchmark": "#7f7f7f",  # 基准 - 灰
    "drawdown": "#d62728",  # 回撤 - 红
    "profit": "#2ca02c",  # 盈利 - 绿
    "loss": "#d62728",  # 亏损 - 红
    "sharpe": "#ff7f0e",  # 夏普 - 橙
    "heatmap_cmap": "RdYlGn",  # 热力图配色
    "scatter_profit": "#2ca02c",
    "scatter_loss": "#d62728",
}

# ── 守护 ──────────────────────────────────────────────

FIG_DPI = 120
FONT_FAMILY = "sans-serif"


def _require_matplotlib():
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib 未安装。生成图表需要: pip install matplotlib")


# ── 全局样式 ──────────────────────────────────────────

FIG_DPI = 120
FONT_FAMILY = "SimHei"


def _require_matplotlib():
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib 未安装。生成图表需要: pip install matplotlib")


def set_style() -> None:
    """全局 matplotlib 样式 — 配置中文字体。"""
    _require_matplotlib()
    matplotlib.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,  # 中文环境下负号正常显示
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def new_figure(nrows: int = 1, ncols: int = 1, figsize: tuple = (12, 6)) -> tuple:
    """创建新图，返回 (fig, axes)。"""
    _require_matplotlib()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def percent_formatter(ax) -> None:
    """Y 轴百分比格式化。"""
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))


def money_formatter(ax) -> None:
    """Y 轴金额格式化（万元）。"""

    def _fmt(x, _):
        if abs(x) >= 1e4:
            return f"{x / 1e4:.0f}万"
        return f"{x:.0f}"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt))


def save_and_close(fig, path: str) -> None:
    """保存图片并关闭。"""
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
