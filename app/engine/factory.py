"""引擎工厂 — 统一创建回测/模拟/实盘环境。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class FactoryConfig:
    """引擎工厂配置。"""

    env_type: str
    config_path: str | None = None
    config_dict: dict | None = None


class EngineFactory:
    """交易引擎工厂。

    用法:
        # 回测
        env = EngineFactory.create("backtest", catalog=catalog, portfolio=portfolio)

        # 模拟交易
        env = EngineFactory.create("paper", catalog=catalog, portfolio=portfolio)

        # 实盘
        env = EngineFactory.create("live", client=client, portfolio=portfolio)
    """

    @classmethod
    def create(
        cls,
        env_type: str,
        **kwargs: object,
    ) -> object:
        """创建交易环境。

        参数:
            env_type: 环境类型 ("backtest", "paper", "live")
            **kwargs: 传递给环境的参数

        返回:
            初始化后的环境实例

        异常:
            ValueError: 未知环境类型
        """
        if env_type == "backtest":
            return cls._create_backtest(**kwargs)
        if env_type == "paper":
            return cls._create_paper(**kwargs)
        if env_type == "live":
            return cls._create_live(**kwargs)

        raise ValueError(
            f"Unknown environment type: {env_type}. "
            f"Available: ['backtest', 'paper', 'live']"
        )

    @classmethod
    def _create_backtest(cls, **kwargs: object) -> object:
        """创建回测环境。"""
        from app.engine.backtest.engine import Engine

        return Engine(**kwargs)  # type: ignore[arg-type] — **kwargs 动态匹配 Engine 构造函数签名

    @classmethod
    def _create_paper(cls, **kwargs: object) -> object:
        """创建模拟交易环境。"""
        from app.engine.paper.environment import PaperEnvironment

        return PaperEnvironment(**kwargs)  # type: ignore[arg-type] — **kwargs 动态匹配 PaperEnvironment 构造函数签名

    @classmethod
    def _create_live(cls, **kwargs: object) -> object:
        """创建实盘交易环境。"""
        from app.engine.live.engine import LiveEnvironment

        return LiveEnvironment(**kwargs)  # type: ignore[arg-type] — **kwargs 动态匹配 LiveEnvironment 构造函数签名

    @classmethod
    def available_environments(cls) -> list[str]:
        """获取可用环境类型列表。"""
        return ["backtest", "paper", "live"]
