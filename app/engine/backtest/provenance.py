"""运行溯源：记录「这个数字跑在哪份数据、哪份代码上」。

背景（2026-09-16）：`data/market/1d` 的全部 5,452 个 parquet 在 22:51 被整体
重建，而 v10 的 34 轮结果全部产出在重建之前。同一个 R32 配置，重建前记录
年化 23.24%、重建后实测 15.36%（-7.9pp）；只翻结果目录里的数字，无法分辨
哪个跑在哪份数据上。因此每次运行都把数据副本指纹与代码状态写进
metadata.json —— 复核数字之前先比指纹。

指纹口径说明：

- ``tree_digest``：对「相对路径 + 文件大小」求 sha256，不做内容哈希（行情
  数百 MB，逐 run 读取不可接受）。同一份副本重复运行得到同一指纹；副本被
  重建、增删标的或数值改写导致压缩后大小变化时，指纹改变。
- ``index_digest``：指数桶（文件小）额外做内容哈希，用来捕捉「文件数与大小
  都不变、但数值被改写」的情况。
- ``code``：git 提交号与工作区是否脏。脏工作区意味着结果不可复现，需在
  归档时说明。
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.settings import get_config

BAR_GLOB = "*.parquet"
PERIOD_DIR = "1d"
INDEX_BUCKET = "IDX"
GIT_TIMEOUT_SECONDS = 5
DIGEST_PREFIX_LENGTH = 16


def _iter_bar_files(root: Path) -> list[tuple[str, int, float]]:
    """列出副本内全部 bar 文件（相对路径、字节数、修改时间），按路径排序。"""
    files: list[tuple[str, int, float]] = []
    for path in root.rglob(BAR_GLOB):
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime))
    files.sort()
    return files


def _digest(lines: list[str]) -> str:
    """对若干文本行求 sha256 前缀摘要。"""
    sha = hashlib.sha256()
    for line in lines:
        sha.update(line.encode("utf-8"))
        sha.update(b"\n")
    return sha.hexdigest()[:DIGEST_PREFIX_LENGTH]


def _dir_content_digest(bucket: Path) -> str | None:
    """目录内文件内容摘要（文件不存在时返回 None）。"""
    if not bucket.is_dir():
        return None
    sha = hashlib.sha256()
    for path in sorted(bucket.rglob(BAR_GLOB)):
        try:
            sha.update(path.relative_to(bucket).as_posix().encode("utf-8"))
            sha.update(path.read_bytes())
        except OSError:
            continue
    return sha.hexdigest()[:DIGEST_PREFIX_LENGTH]


def market_fingerprint(market_dir: Path) -> dict[str, Any]:
    """市场副本指纹（标的数、总字节、树摘要、指数桶内容摘要）。"""
    if not market_dir.is_dir():
        return {"available": False, "path": str(market_dir)}
    files = _iter_bar_files(market_dir)
    if not files:
        return {"available": False, "path": str(market_dir), "files": 0}
    latest = max(mtime for _, _, mtime in files)
    return {
        "available": True,
        "path": str(market_dir),
        "files": len(files),
        "total_bytes": sum(size for _, size, _ in files),
        "latest_mtime": datetime.fromtimestamp(latest).isoformat(timespec="seconds"),
        "tree_digest": _digest([f"{rel}|{size}" for rel, size, _ in files]),
        "index_digest": _dir_content_digest(market_dir / INDEX_BUCKET),
    }


def _git(repo_root: Path, *args: str) -> str | None:
    """执行 git 子命令，失败返回 None（不抛异常，溯源不能拖垮回测）。"""
    try:
        proc = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    # stdout 可能为 None：子进程输出含非本地编码（gbk）字节时，读取线程解码失败
    return (proc.stdout or "").strip()


def code_state(repo_root: Path) -> dict[str, Any]:
    """代码状态（提交号、分支、工作区是否脏）。"""
    commit = _git(repo_root, "rev-parse", "--short", "HEAD")
    if commit is None:
        return {"available": False}
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    return {
        "available": True,
        "commit": commit,
        "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or "",
        "dirty": bool(status),
    }


def repo_root() -> Path:
    """项目根目录（app/engine/backtest/provenance.py → 上溯三级）。"""
    return Path(__file__).resolve().parents[3]


def collect_provenance() -> dict[str, Any]:
    """汇总本次运行的市场副本指纹与代码状态。"""
    market_root = Path(get_config().data.market_path)
    return {
        "market": market_fingerprint(market_root / PERIOD_DIR),
        "code": code_state(repo_root()),
    }
