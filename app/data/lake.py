"""数据湖（qdata）只读访问：根目录解析与年份分区读取。

湖根目录来源优先级：环境变量 ``QDATA_ROOT`` → ``market-data/.env``。
代码中禁止硬编码湖路径；消费方只读 ``clean/`` 与 ``meta/``，不直读 ``raw/``。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ENV_KEY = "QDATA_ROOT"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT.parent / "market-data" / ".env"
CLEAN_LAYER = "clean"
META_LAYER = "meta"
YEAR_PARTITION_GLOB = "year=*/*.parquet"


def resolve_data_root() -> Path:
    """解析湖根目录（环境变量优先，其次 market-data/.env）。"""
    value = os.environ.get(ENV_KEY, "").strip()
    if not value and ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
        value = os.environ.get(ENV_KEY, "").strip()
    if not value:
        raise RuntimeError(f"未找到 {ENV_KEY}（环境变量或 {ENV_FILE}）")
    return Path(value).expanduser()


def dataset_dir(dataset: str, layer: str = CLEAN_LAYER) -> Path:
    """返回湖内某数据集目录（不做存在性校验）。"""
    return resolve_data_root() / layer / dataset


def read_year_partitions(
    dataset: str,
    columns: list[str] | None = None,
    layer: str = CLEAN_LAYER,
) -> pd.DataFrame:
    """读取某数据集的全部年份分区并拼接。

    Args:
        dataset: 数据集名（如 ``financials``）
        columns: 需要的列；None 表示全部
        layer: 湖分层（默认 ``clean``）

    Returns:
        拼接后的长表 DataFrame

    Raises:
        FileNotFoundError: 数据集目录下无年份分区
        RuntimeError: 某个分区读取失败
    """
    root = dataset_dir(dataset, layer)
    paths = sorted(root.glob(YEAR_PARTITION_GLOB))
    if not paths:
        raise FileNotFoundError(f"未找到 {dataset} 分区：{root}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path, columns=columns))
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"读取湖分区失败：{path}") from exc
    return pd.concat(frames, ignore_index=True)
