"""因子模块。

子模块：
- base/: 因子基类、注册表、运算符
- universe.py: 股票池管理
- engine.py: 因子计算引擎
- precomputed.py: 技术指标预计算缓存
- store.py: 因子存储与 O(1) 查找
- processor.py: 因子预处理（去极值、标准化、中性化）
"""

from app.factors.base import (
    Factor,
    FactorCategory,
    DataBundle,
    FactorMetadata,
    canonical_name,
    FactorRegistry,
    register_factor,
    clear_registry,
    FactorDAG,
    Ops,
    CrossSectionOps,
    NeutralizeMethod,
)
from app.factors.universe import Universe, UniverseManager, UniverseRecord
from app.factors.engine import FactorEngine
from app.factors.precomputed import (
    PrecomputedIndicators,
    PrecomputedIndicatorCache,
    build_precomputed_cache,
)
from app.factors.store import FactorMatrix, FactorStore, CachingFactorStore
from app.factors.processor import FactorProcessor
from app.factors.indicator import IndicatorCalculator

__all__ = [
    "Factor",
    "FactorCategory",
    "DataBundle",
    "FactorMetadata",
    "canonical_name",
    "FactorRegistry",
    "register_factor",
    "clear_registry",
    "FactorDAG",
    "Ops",
    "CrossSectionOps",
    "NeutralizeMethod",
    "Universe",
    "UniverseManager",
    "UniverseRecord",
    "FactorEngine",
    "PrecomputedIndicators",
    "PrecomputedIndicatorCache",
    "build_precomputed_cache",
    "FactorMatrix",
    "FactorStore",
    "CachingFactorStore",
    "FactorProcessor",
    "IndicatorCalculator",
]
