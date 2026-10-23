"""配置层核心模块。

负责加载、解析、验证项目配置，提供全局单例访问接口。
配置来源：config.toml（默认值）+ 环境变量（覆盖）。
"""

import logging
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import tomllib

logger = logging.getLogger(__name__)

# 配置文件路径（与 settings.py 同级目录）
CONFIG_FILE = Path(__file__).parent / "config.toml"

# 全局单例缓存
_config_instance = None


# ─── 配置数据类 ────────────────────────────────────────────────


@dataclass
class ProjectConfig:
    """项目基础配置。"""

    name: str = "plum"
    env: str = "dev"
    debug: bool = False


@dataclass
class DataConfig:
    """数据源配置 — 所有路径统一在 data/ 下，均为 Path 对象。

    派生文件路径通过属性访问（如 cfg.data.delisting_registry_file）。
    """

    db_path: Path = Path("./data")
    """根数据目录 — 所有数据存储的统一根路径。"""

    market_path: Path = Path("./data/market")
    """K 线数据路径 — parquet 文件 + meta.json。"""

    financial_path: Path = Path("./data/financial")
    """财务数据路径 — 利润表/资产负债表/现金流量表。"""

    results_path: Path = Path("./data/results")
    """回测结果路径 — 报告/图表/权益曲线。"""

    factors_path: Path = Path("./data/factors")
    """因子缓存路径 — 预计算因子 parquet + TieredStorage。"""

    charts_path: Path = Path("./data/charts")
    """图表输出路径 — 回测可视化 PNG/SVG。"""

    registry_path: Path = Path("./data/registry")
    """注册表路径 — 退市记录/股票生命周期。"""

    ledger_path: Path = Path("./data/ledger")
    """交易账本路径 — 实盘交易记录。"""

    cache_path: Path = Path("./data/cache")
    """缓存根目录 — storage 元数据 + factor universe 快照。"""

    benchmarks_path: Path = Path("./data/benchmarks")
    """基准指数路径 — plum 自建风格指数的发布产物（CSV/parquet）。"""

    storage_cache_path: Path = Path("./data/cache/storage")
    """Storage 元数据缓存路径 — sync_progress.json, meta.json。"""

    factor_universe_cache_path: Path = Path("./data/cache/factors/universe")
    """因子 universe 快照缓存路径。"""

    bar_format: str = "parquet"
    validate_on_load: bool = True

    # ── 派生文件路径（属性） ──────────────────────────

    @property
    def delisting_registry_file(self) -> Path:
        """退市注册表 JSON 文件路径。"""
        return self.registry_path / "delisting_registry.json"

    @property
    def storage_meta_file(self) -> Path:
        """Storage 元数据 JSON 文件路径。"""
        return self.storage_cache_path / "meta.json"

    @property
    def trade_ledger_file(self) -> Path:
        """实盘交易账本 SQLite 文件路径。"""
        return self.ledger_path / "trade_ledger.sqlite"


@dataclass
class CostConfig:
    """交易成本配置（影响回测真实性）。

    金额和费率使用 Decimal 避免浮点精度问题。
    """

    commission_rate: Decimal = field(default_factory=lambda: Decimal("0.00025"))
    min_commission: Decimal = field(default_factory=lambda: Decimal("5.0"))
    stamp_tax: Decimal = field(default_factory=lambda: Decimal("0.0005"))
    transfer_fee: Decimal = field(default_factory=lambda: Decimal("0.00001"))
    slippage: Decimal = field(default_factory=lambda: Decimal("0.0"))


@dataclass
class RiskConfig:
    """风控与限制配置。"""

    max_position_pct: float = 0.1
    max_drawdown: float = 0.2
    limit_up_deal: bool = False
    limit_down_deal: bool = False


@dataclass
class RuleConfig:
    """规则模块配置。"""

    cache_enabled: bool = True


@dataclass
class EngineConfig:
    """回测引擎配置。"""

    backtest_start: str = "2020-01-01"
    backtest_end: str = "2024-12-31"
    initial_capital: int = 1_000_000


@dataclass
class StrategyConfig:
    """策略模块配置。"""

    default: str = "buy_and_hold"


@dataclass
class LoggingConfig:
    """日志配置。"""

    level: str = "DEBUG"
    format: str = "rich"


@dataclass
class BrokerConfig:
    """券商/交易网关配置。

    QMT 敏感信息（qmt_path, account_id）通过环境变量覆盖，
    不在 config.toml 中硬编码。
    """

    provider: str = "qmt"
    qmt_path: str = ""
    account_id: str = ""
    session_id: int = 9527
    trade_env: str = "sim"  # sim=模拟盘, real=实盘


@dataclass
class Settings:
    """配置总入口，聚合所有子配置。"""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    rule: RuleConfig = field(default_factory=RuleConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)


# ─── 内部工具函数 ──────────────────────────────────────────────


def _load_toml(path: Path) -> dict:
    """读取并解析 TOML 配置文件。"""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"配置文件不可读或非法 TOML: {path}（{exc}）") from exc


def _apply_env_overrides(raw: dict) -> dict:
    """用环境变量覆盖配置。

    命名规则：PLUM_<SECTION>_<KEY>
    示例：PLUM_DATA_DB_PATH=/custom/path → data.db_path = "/custom/path"
    环境变量优先级高于 TOML 文件。
    """
    prefix = "PLUM_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix) :].lower().split("_", 1)
            if len(parts) == 2:
                section, field_name = parts
                if section in raw:
                    raw[section][field_name] = _cast_value(value)
    return raw


def _cast_value(value: str):
    """将字符串值转换为合适的 Python 类型。

    优先级：bool → int（≤10位纯数字） → float → str
    超长数字串（>10位，如账户ID）保留为字符串。
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    # 纯数字且合理长度才转为 int（避免账户ID被误转）
    if value.isdigit() and len(value) <= 10:
        try:
            return int(value)
        except ValueError:  # pragma: no cover — isdigit 已排除非数字
            return value
    # 纯数字但过长 → 保留为字符串（账户ID场景）
    if value.isdigit():
        return value
    try:
        return float(value)
    except ValueError:
        logger.debug("配置值类型转换失败，保留原始值", exc_info=True)
    return value


def _dict_to_dataclass(cls, data: dict):
    """将字典映射到 dataclass，忽略未知字段，缺失字段使用默认值。"""
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in fields}
    return cls(**filtered)


def _build_settings(raw: dict) -> Settings:
    """将原始字典构建为 Settings 对象。"""
    cost_raw = raw.get("cost", {})
    cost_data = {
        k: Decimal(str(v)) if isinstance(v, (int, float)) else v
        for k, v in cost_raw.items()
    }
    data_raw = raw.get("data", {})
    data_converted = {
        k: Path(v) if k.endswith("_path") and isinstance(v, str) else v
        for k, v in data_raw.items()
    }
    return Settings(
        project=_dict_to_dataclass(ProjectConfig, raw.get("project", {})),
        data=_dict_to_dataclass(DataConfig, data_converted),
        cost=_dict_to_dataclass(CostConfig, cost_data),
        risk=_dict_to_dataclass(RiskConfig, raw.get("risk", {})),
        rule=_dict_to_dataclass(RuleConfig, raw.get("rule", {})),
        engine=_dict_to_dataclass(EngineConfig, raw.get("engine", {})),
        strategy=_dict_to_dataclass(StrategyConfig, raw.get("strategy", {})),
        logging=_dict_to_dataclass(LoggingConfig, raw.get("logging", {})),
        broker=_dict_to_dataclass(BrokerConfig, raw.get("broker", {})),
    )


# ─── 公开接口 ──────────────────────────────────────────────────


def validate_settings(settings: Settings) -> None:
    """验证配置合法性，失败则抛出 ValueError。

    校验项：
    - 数据路径不能为空
    - 初始资金必须大于 0
    - 佣金率必须在 0~1 之间
    - 单票持仓占比必须在 0~1 之间
    - 环境标识必须为 dev/staging/prod
    - 日志级别必须有效
    - prod 环境强制 broker 配置完整
    """
    errors: list[str] = []

    # ── 数据路径 ──────────────────────────────────
    if not settings.data.db_path:
        errors.append("data.db_path 不能为空")

    # ── 引擎参数 ──────────────────────────────────
    if settings.engine.initial_capital <= 0:
        errors.append("engine.initial_capital 必须大于 0")

    # ── 费率范围 ──────────────────────────────────
    if not (Decimal("0") < settings.cost.commission_rate < Decimal("1")):
        errors.append("cost.commission_rate 必须在 0~1 之间")
    if settings.cost.min_commission < Decimal("0"):
        errors.append("cost.min_commission 不能为负数")
    if not (Decimal("0") <= settings.cost.stamp_tax < Decimal("1")):
        errors.append("cost.stamp_tax 必须在 0~1 之间")

    # ── 风控范围 ──────────────────────────────────
    if not (Decimal("0") < settings.risk.max_position_pct <= Decimal("1")):
        errors.append("risk.max_position_pct 必须在 0~1 之间")
    if not (Decimal("0") <= settings.risk.max_drawdown <= Decimal("1")):
        errors.append("risk.max_drawdown 必须在 0~1 之间")

    # ── 环境合法性 ────────────────────────────────
    valid_envs = {"dev", "staging", "prod"}
    if settings.project.env not in valid_envs:
        errors.append(
            f"project.env 必须为 {'/'.join(sorted(valid_envs))}，"
            f"当前值: {settings.project.env}"
        )

    # ── 日志级别 ──────────────────────────────────
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if settings.logging.level.upper() not in valid_levels:
        errors.append(
            f"logging.level 无效: {settings.logging.level}，"
            f"有效值: {', '.join(sorted(valid_levels))}"
        )

    # ── prod 环境强制 broker 配置 ────────────────
    if settings.project.env == "prod":
        if not settings.broker.qmt_path:
            errors.append("prod 环境必须配置 broker.qmt_path")
        if not settings.broker.account_id:
            errors.append(
                "prod 环境必须配置 broker.account_id (可通过 PLUM_BROKER_ACCOUNT_ID 环境变量设置)"
            )

    if errors:
        raise ValueError("\n  - " + "\n  - ".join(errors))


def get_config() -> Settings:
    """获取全局配置单例。

    首次调用时加载配置并验证，后续调用直接返回缓存实例。
    """
    global _config_instance
    if _config_instance is None:
        raw = _load_toml(CONFIG_FILE)
        raw = _apply_env_overrides(raw)
        _config_instance = _build_settings(raw)
        validate_settings(_config_instance)
    return _config_instance


def reset_config() -> None:
    """重置配置单例（主要用于测试）。"""
    global _config_instance
    _config_instance = None
