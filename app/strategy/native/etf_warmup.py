"""ETF 策略预热：把回测起点之前的历史预载进策略缓存（严格 PIT，不含未来信息）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from app.strategy.native.etf_common import CORR_WINDOW, _num


class EtfWarmupMixin:
    """为 ETF 策略预载起点前历史，并提供建仓所需的最少长度。

    用法：策略类继承本 mixin，在 ``__init__`` 末尾按需调用
    ``self._load_warmup(warmup_until)``；子类可覆写 ``_load_warmup``
    做定向载入（如只读 basket 内标的）。
    """

    momentum_window: int
    ma_window: int
    _adj_close: dict[str, list[float]]
    _amount: dict[str, list[float]]

    def _warmup(self) -> int:
        """建仓所需最少历史长度（动量窗与均线窗取大者，再加 1）。"""
        return max(self.momentum_window + 1, self.ma_window + 1)

    def _load_warmup(self, until: datetime) -> None:
        """预载全市场副本中 ``until`` 之前的历史。

        引擎只把 ``[start, end]`` 的 K 线喂给策略，不预载的话头 ``_warmup()`` 个
        交易日会因历史不足而空仓，与因子层能直接取到起点前历史的股票策略不可比。
        只读取 ``until`` 之前的行（严格 PIT，不引入未来信息）。
        """
        root = self._market_root()
        if root is None:
            return
        cutoff = pd.Timestamp(until) - pd.Timedelta(days=1)
        keep = self._warmup() + CORR_WINDOW
        for directory in ("SH", "SZ"):
            for path in sorted((root / directory).glob("*.parquet")):
                frame = pd.read_parquet(path, columns=["close", "adj_factor", "amount"])
                frame = frame[frame.index <= cutoff].tail(keep)
                if frame.empty:
                    continue
                symbol = f"{directory}.{path.stem}"
                prices = frame["close"].astype("float64") * frame["adj_factor"].astype(
                    "float64"
                )
                self._adj_close[symbol] = [_num(v) for v in prices]
                self._amount[symbol] = [
                    _num(v) for v in frame["amount"].astype("float64")
                ]

    @staticmethod
    def _market_root() -> Path | None:
        """定位 plum 市场副本的日线目录（供预热读取）。"""
        try:
            from app.config.settings import get_config

            root = Path(str(get_config().data.market_path)) / "1d"
        except Exception:  # 配置不可用时静默降级：不预热也能跑，只是开头空仓
            return None
        return root if root.is_dir() else None
