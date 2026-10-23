"""因子注册表 — 支持自动注册。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from app.factors.base.factor import Factor, FactorCategory

if TYPE_CHECKING:
    from app.factors.base.factor import FactorMetadata

logger = logging.getLogger(__name__)

FactorDecorator = Callable[[type[Factor]], type[Factor]]


class FactorRegistry:
    """因子中心注册表。

    单例模式 — 使用 FactorRegistry.get_instance() 进行线程安全访问。

    特性：
    - 通过 @register_factor 装饰器自动注册
    - 按规范名称查找
    - 按分类过滤
    - 每个因子的元数据存储
    - 持久化到 YAML/JSON
    """

    _instance: FactorRegistry | None = None

    def __init__(self) -> None:
        self._factors: dict[str, type[Factor]] = {}
        self._metadata: dict[str, FactorMetadata] = {}
        self._category_index: dict[FactorCategory, list[str]] = {
            cat: [] for cat in FactorCategory
        }

    @classmethod
    def get_instance(cls) -> FactorRegistry:
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, factor_cls: type[Factor]) -> None:
        """注册因子类。"""
        name = factor_cls.name

        if name in self._factors:
            logger.warning(f"因子 {name} 已注册，跳过")
            return

        if not name:
            raise ValueError(f"因子类 {factor_cls.__name__} 缺少 'name' 属性")

        self._factors[name] = factor_cls
        self._category_index[factor_cls.category].append(name)

        instance = factor_cls()
        self._metadata[name] = instance.get_metadata()

        logger.info(f"已注册因子: {name} ({factor_cls.category.value})")

    def unregister(self, name: str) -> bool:
        """注销因子。"""
        if name not in self._factors:
            return False

        factor_cls = self._factors[name]
        self._category_index[factor_cls.category].remove(name)
        del self._factors[name]
        self._metadata.pop(name, None)

        logger.info(f"已注销因子: {name}")
        return True

    def get(self, name: str) -> type[Factor] | None:
        """按名称获取因子类。"""
        return self._factors.get(name)

    def get_factor_instance(self, name: str) -> Factor | None:
        """按名称获取因子实例。"""
        cls = self._factors.get(name)
        if cls is None:
            return None
        return cls()

    def list_all(self) -> list[str]:
        """列出所有已注册的因子名称。"""
        return list(self._factors.keys())

    def list_by_category(self, category: FactorCategory) -> list[str]:
        """列出某个分类下的因子名称。"""
        return list(self._category_index[category])

    def get_metadata(self, name: str) -> FactorMetadata | None:
        """获取因子元数据。"""
        return self._metadata.get(name)

    def get_all_metadata(self) -> dict[str, FactorMetadata]:
        """获取所有因子元数据。"""
        return dict(self._metadata)

    def save_metadata(self, path: Path | str) -> None:
        """保存元数据到 YAML 文件。"""
        data = {
            name: {
                "name": meta.name,
                "category": meta.category,
                "indicator": meta.indicator,
                "period": meta.period,
                "processing": meta.processing,
                "description": meta.description,
                "dependencies": meta.dependencies,
                "data_source": meta.data_source,
                "version": meta.version,
                "last_updated": meta.last_updated.isoformat()
                if meta.last_updated
                else None,
            }
            for name, meta in self._metadata.items()
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load_metadata(self, path: Path | str) -> None:
        """从 YAML 文件加载元数据。"""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for name, meta_dict in data.items():
            from app.factors.base.factor import FactorMetadata

            meta = FactorMetadata(
                name=meta_dict["name"],
                category=meta_dict["category"],
                indicator=meta_dict.get("indicator", ""),
                period=meta_dict.get("period", ""),
                processing=meta_dict.get("processing", ""),
                description=meta_dict.get("description", ""),
                dependencies=meta_dict.get("dependencies", []),
                data_source=meta_dict.get("data_source", ""),
                version=meta_dict.get("version", "1.0.0"),
            )
            if meta_dict.get("last_updated"):
                meta.last_updated = datetime.fromisoformat(meta_dict["last_updated"])

            self._metadata[name] = meta

    @property
    def count(self) -> int:
        """已注册因子总数。"""
        return len(self._factors)

    def summary(self) -> pd.DataFrame:
        """获取所有因子摘要。"""
        rows = []
        for name, meta in self._metadata.items():
            rows.append(
                {
                    "name": name,
                    "category": meta.category,
                    "indicator": meta.indicator,
                    "period": meta.period,
                    "processing": meta.processing,
                    "dependencies": len(meta.dependencies),
                    "data_source": meta.data_source,
                }
            )
        return pd.DataFrame(rows).sort_values("name")


def _get_registry() -> FactorRegistry:
    return FactorRegistry.get_instance()


def register_factor(cls: type[Factor]) -> type[Factor]:
    """装饰器：自动注册因子类。

    用法:
        @register_factor
        class VAL_PE_TTM_ZSCORE(Factor):
            name = "VAL_PE_TTM_ZSCORE"
            category = FactorCategory.VALUATION
            ...
    """
    registry = _get_registry()
    registry.register(cls)
    return cls


def clear_registry() -> None:
    """清空注册表。用于测试。"""
    FactorRegistry._instance = None


class FactorDAG:
    """因子依赖解析的有向无环图。"""

    def __init__(self, registry: FactorRegistry) -> None:
        self._registry = registry
        self._adjacency: dict[str, set[str]] = {}
        self._in_degree: dict[str, int] = {}

    def build(self, factor_names: list[str]) -> None:
        """构建依赖图。"""
        self._adjacency.clear()
        self._in_degree.clear()

        for name in factor_names:
            self._adjacency[name] = set()
            self._in_degree[name] = 0

        for name in factor_names:
            factor_cls = self._registry.get(name)
            if factor_cls is None:
                continue
            instance = factor_cls()
            for dep in instance.dependencies:
                if dep in self._adjacency:
                    self._adjacency[dep].add(name)
                    self._in_degree[name] += 1

    def topological_sort(self) -> list[str]:
        """拓扑排序。"""
        queue: list[str] = [n for n, d in self._in_degree.items() if d == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in self._adjacency[node]:
                self._in_degree[neighbor] -= 1
                if self._in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def detect_cycle(self) -> list[str] | None:
        """检测循环依赖。"""
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> list[str] | None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._adjacency[node]:
                if neighbor not in visited:
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            path.pop()
            rec_stack.remove(node)
            return None

        for node in self._adjacency:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle
        return None

    def get_computation_order(self, factor_names: list[str]) -> list[str]:
        """获取计算顺序。"""
        self.build(factor_names)
        cycle = self.detect_cycle()
        if cycle:
            raise ValueError(f"检测到循环依赖: {' -> '.join(cycle)}")
        return self.topological_sort()

    def get_dependencies(self, factor_name: str) -> set[str]:
        """获取因子的所有依赖。"""
        visited: set[str] = set()
        deps: set[str] = set()

        def dfs(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            factor_cls = self._registry.get(name)
            if factor_cls is None:
                return
            instance = factor_cls()
            for dep in instance.dependencies:
                deps.add(dep)
                dfs(dep)

        dfs(factor_name)
        return deps

    def get_independent_factors(self, factor_names: list[str]) -> list[str]:
        """获取无依赖的独立因子。"""
        result: list[str] = []
        for name in factor_names:
            factor_cls = self._registry.get(name)
            if factor_cls is None:
                continue
            instance = factor_cls()
            has_dep_in_list = any(dep in factor_names for dep in instance.dependencies)
            if not has_dep_in_list:
                result.append(name)
        return result
