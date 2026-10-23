"""项目统一的日志初始化 — 替代所有散落的 basicConfig 调用。

用法:
    from app.config.settings import get_config
    from app.config.logging import setup_logging

    settings = get_config()
    setup_logging(settings)  # 一次调用，全局生效
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.settings import Settings

__all__ = ["setup_logging", "RunContext", "start_run"]


class RunContext(logging.Filter):
    """给每条日志注入 run_id，用于按运行回放日志。"""

    run_id: str = ""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id or "-"
        return True


_run_context = RunContext()


def setup_logging(settings: Settings) -> None:
    """一次性初始化日志系统。

    替代 main.py / multi_runner.py 中的 logging.basicConfig 调用。
    - dev/staging: 仅 stderr 输出
    - prod: 额外 JSON 格式文件输出到 data/logs/plum.log
    """
    level_name = settings.logging.level.upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # ── stderr handler（所有环境） ──────────────────
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(stderr_handler)

    # ── file handler（仅 prod） ────────────────────
    if settings.project.env == "prod":
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "plum.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                '{"time":"%(asctime)s","level":"%(levelname)s",'
                '"logger":"%(name)s","run_id":"%(run_id)s",'
                '"message":"%(message)s"}'
            )
        )
        root.addHandler(file_handler)

    # ── 注入 run_id ───────────────────────────────
    root.addFilter(_run_context)

    # ── 抑制第三方库的 DEBUG 噪音 ─────────────────
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def start_run(run_id: str | None = None) -> str:
    """开始一个新运行，返回 run_id（同时注入到所有后续日志）。

    调用:
        from app.config.logging import start_run
        rid = start_run()  # 自动生成 UUID
        # 或
        rid = start_run("backtest-20260528-001")  # 指定 ID
    """
    import uuid

    _run_context.run_id = run_id or uuid.uuid4().hex[:12]
    return _run_context.run_id
