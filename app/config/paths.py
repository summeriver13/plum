"""路径常量 — 从统一配置 DataConfig 导出，向前兼容旧 config.paths API。

所有数据路径由 app/config/config.toml 统一管理。
本模块提供模块级 Path 常量，等价于旧 config/paths.py 的接口。

用法:
    from app.config.paths import MARKET_PATH, RESULTS_PATH, ...
"""

from app.config.settings import get_config

_cfg = get_config().data

# ── 一级数据目录 ────────────────────────────────────────

DATA_ROOT = _cfg.db_path

MARKET_PATH = _cfg.market_path
"""K 线数据: parquet 文件 + meta.json。"""

FINANCIAL_PATH = _cfg.financial_path
"""财务数据: 利润表/资产负债表/现金流量表/每股指标。"""

RESULTS_PATH = _cfg.results_path
"""回测结果: 报告/图表/权益曲线/交易日志。"""

FACTORS_PATH = _cfg.factors_path
"""因子缓存: 预计算因子 parquet + TieredStorage。"""

CHARTS_PATH = _cfg.charts_path
"""图表输出: 回测可视化 PNG/SVG。"""

REGISTRY_PATH = _cfg.registry_path
"""注册表: 退市记录、股票生命周期元数据。"""

LEDGER_PATH = _cfg.ledger_path
"""交易账本: 实盘交易记录 SQLite。"""

BENCHMARKS_PATH = _cfg.benchmarks_path
"""基准指数: plum 自建风格指数的发布产物（CSV/parquet）。

与 ``data/market/`` 分开存放：market 是数据湖的派生副本，会被重建脚本覆盖；
本目录由 ``scripts/build_microcap_index.py --publish`` 生成。
"""

# ── 二级缓存目录 ────────────────────────────────────────

CACHE_ROOT = _cfg.cache_path
"""缓存根目录。"""

STORAGE_CACHE = _cfg.storage_cache_path
"""Storage 元数据: sync_progress.json, sync_states.jsonl, meta.json。"""

FACTOR_UNIVERSE_CACHE = _cfg.factor_universe_cache_path
"""因子 universe 快照缓存。"""

# ── 具体文件路径 ────────────────────────────────────────

DELISTING_REGISTRY_FILE = _cfg.delisting_registry_file
"""退市注册表 JSON 文件。"""

STORAGE_META_FILE = _cfg.storage_meta_file
"""Storage 元数据 JSON 文件 (stock_names 等)。"""

TRADE_LEDGER_FILE = _cfg.trade_ledger_file
"""实盘交易账本 SQLite 文件。"""
