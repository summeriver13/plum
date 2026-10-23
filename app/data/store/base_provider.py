"""物理存储层 — 仅处理磁盘 I/O 和并发。

提供：
- BaseStorageProvider: 纯存储，无业务逻辑（基于 meta.json）
- _BaseCatalog: 存储 + 注册表的薄封装
- 股票代码/交易所转换工具
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.paths import STORAGE_CACHE


def symbol_to_exchange_code(symbol: str) -> tuple[str, str]:
    """将 Plum 格式股票代码转换为 (交易所, 代码) 元组。

    参数:
        symbol: Plum 格式（如 "600519.SH"、"000001.SZ"、"600519"）
                或 QMT 格式（如 "600519.SH"）

    返回:
        (交易所, 代码) 元组，交易所大写（如 ("SH", "600519")）
    """
    if "." in symbol:
        prefix, suffix = symbol.split(".", 1)
        prefix = prefix.upper()
        if suffix.isalpha():
            return suffix.upper(), prefix
        if prefix in ("SH", "SZ", "BJ"):
            return prefix, suffix
        return prefix, suffix

    code = symbol.upper()
    if code.startswith(("4", "8")):
        return "BJ", code
    if code.startswith(("0", "3", "1")):
        return "SZ", code
    if code.startswith(("5", "6")):
        return "SH", code
    return "SH", code


def exchange_code_to_symbol(exchange: str, code: str) -> str:
    """将 (交易所, 代码) 转换为 Plum 格式股票代码。"""
    return f"{exchange.lower()}.{code}"


class BaseStorageProvider:
    """底层存储引擎：仅处理物理读写。

    特性：
    - Parquet I/O（snappy 压缩）
    - 基于 meta.json 的元数据注册表（线程安全）
    - 数据指纹（行数、哈希）
    - 无业务逻辑 — 纯基础设施
    """

    _meta_lock = threading.Lock()

    def __init__(self, root_dir: Path | str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._meta_cache: dict[str, Any] | None = None
        self._pending_meta: dict[str, Any] = {}

    @staticmethod
    def make_path(*parts: str) -> Path:
        """从路径片段构建路径。"""
        if not parts:
            return Path(".")
        return Path("/".join(str(p) for p in parts))

    def resolve(self, *parts: str) -> Path:
        """解析相对于 root 的路径。"""
        str_parts = [str(p) for p in parts]
        return self.root / self.make_path(*str_parts)

    def read_parquet(self, path: Path) -> pd.DataFrame:
        """从 Parquet 文件读取 DataFrame。"""
        resolved = path if path.is_absolute() else self.root.joinpath(path)
        if not resolved.exists():
            return pd.DataFrame()
        return pd.read_parquet(resolved)

    def write_parquet(
        self,
        df: pd.DataFrame,
        path: Path,
        mode: str = "overwrite",
        dedup_column: str | None = None,
    ) -> None:
        """写入 DataFrame 到 Parquet 文件。"""
        if not path.is_absolute():
            root_name = self.root.name
            if path.parts and path.parts[0] == root_name:
                pass
            else:
                path = self.resolve(path)

        if df.empty:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "append" and path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df])
            if dedup_column:
                if dedup_column in combined.columns:
                    combined = combined.drop_duplicates(
                        subset=[dedup_column], keep="last"
                    )
                elif dedup_column == combined.index.name:
                    combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(path, compression="snappy")
        else:
            df.to_parquet(path, compression="snappy")

    def delete(self, path: Path) -> bool:
        """删除文件。"""
        if not path.is_absolute():
            path = self.resolve(path)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, path: Path) -> bool:
        """检查路径是否存在。"""
        if not path.is_absolute():
            path = self.resolve(path)
        return path.exists()

    def list_files(self, pattern: str = "*.parquet") -> list[Path]:
        """列出匹配模式的文件。"""
        return sorted(self.root.glob(pattern))

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    def load_meta(self) -> dict[str, Any]:
        """加载元数据（线程安全，缓存）。"""
        if self._meta_cache is not None:
            return self._meta_cache.copy()

        with self._lock:
            if self._meta_cache is not None:
                return self._meta_cache.copy()

            if self.meta_path.exists():
                with open(self.meta_path, encoding="utf-8") as f:
                    self._meta_cache = json.load(f)
            else:
                self._meta_cache = {}
            return self._meta_cache.copy()

    def save_meta(self, meta: dict[str, Any]) -> None:
        """保存元数据（线程安全）。"""
        with self._lock:
            self._save_meta_unsafe(meta)

    def get_entry(self, key: str) -> dict[str, Any] | None:
        """获取单个元数据条目。"""
        return self.load_meta().get(key)

    def set_entry(self, key: str, entry: dict[str, Any]) -> None:
        """设置单个元数据条目（批量 — 同步结束时刷新）。"""
        with self._lock:
            if self._meta_cache is None:
                self._meta_cache = {}
            self._meta_cache[key] = entry
            self._pending_meta[key] = entry

    def flush_meta(self) -> None:
        """刷新所有待处理的元数据条目到 meta.json。"""
        with self._lock:
            if not self._pending_meta:
                return
            current = self.load_meta()
            current.update(self._pending_meta)
            self._save_meta_unsafe(current)
            self._pending_meta.clear()

    def _save_meta_unsafe(self, meta: dict[str, Any]) -> None:
        """写入 meta.json（假设已持有锁）。"""
        self._meta_cache = meta.copy()
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def remove_entry(self, key: str) -> None:
        """移除单个元数据条目。"""
        meta = self.load_meta()
        if key in meta:
            del meta[key]
            self.save_meta(meta)

    def invalidate_meta_cache(self) -> None:
        """使元数据缓存失效，强制下次访问时重新加载。"""
        with self._lock:
            self._meta_cache = None

    @staticmethod
    def compute_fingerprint(df: pd.DataFrame) -> str:
        """计算数据指纹。"""
        if df.empty:
            return "empty"
        first_val = (
            df.iloc[0].to_bytes()
            if hasattr(df.iloc[0], "to_bytes")
            else str(df.iloc[0])
        )
        last_val = (
            df.iloc[-1].to_bytes()
            if hasattr(df.iloc[-1], "to_bytes")
            else str(df.iloc[-1])
        )
        content = f"{len(df)}:{first_val}:{last_val}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def build_entry(
        path: Path,
        df: pd.DataFrame,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """构建标准元数据条目，包含 schema 版本信息。"""
        entry: dict[str, Any] = {
            "path": str(path),
            "count": len(df),
            "fingerprint": BaseStorageProvider.compute_fingerprint(df),
            "updated_at": datetime.now().isoformat(),
        }
        # 从 Parquet attrs 中提取 schema 版本（_bars_to_df 写入）
        if "schema_version" in df.attrs:
            entry["schema_version"] = df.attrs["schema_version"]
        if path.exists():
            entry["size_bytes"] = path.stat().st_size
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            entry["start"] = str(df.index.min())
            entry["end"] = str(df.index.max())
        if extra:
            entry.update(extra)
        return entry


class _BaseCatalog:
    """薄封装，组合存储和注册表层。"""

    ROOT_NAME: str = ""

    def __init__(
        self,
        catalog_root: Path | str | None = None,
        storage: BaseStorageProvider | None = None,
    ) -> None:
        if storage is not None:
            self._storage = storage
        else:
            if catalog_root is None:
                root = STORAGE_CACHE / self.__class__.ROOT_NAME
            else:
                root = Path(catalog_root)
            self._storage = BaseStorageProvider(root)
        self._key_prefix: str = ""

    @property
    def storage(self) -> BaseStorageProvider:
        return self._storage

    @property
    def root(self) -> Path:
        return self._storage.root

    def _make_key(self, symbol: str, *parts: str) -> str:
        """构建元数据键。"""
        parts_str = ":".join(str(p) for p in parts)
        return f"{symbol}:{parts_str}" if parts_str else symbol

    def _symbol_path(self, symbol: str, *parts: str) -> Path:
        """获取股票的文件路径。"""
        exchange, code = symbol_to_exchange_code(symbol)
        parts_str = "/".join(str(p) for p in parts) if parts else ""
        filename = f"{exchange}.{code}"
        if parts_str:
            filename = f"{filename}.{parts_str}"
        return Path(filename)

    def _read(self, path: Path) -> pd.DataFrame:
        """从 Parquet 读取 DataFrame。"""
        path_str = str(path)
        is_resolved = path.is_absolute() or path_str.startswith(
            (f"{self.root.name}/", f"{self.root.name}\\", str(self.root))
        )
        resolved = self.storage.resolve(path) if not is_resolved else path
        if not resolved.exists():
            cwd_resolved = Path.cwd() / resolved
            if not cwd_resolved.exists():
                return pd.DataFrame()
            resolved = cwd_resolved
        return pd.read_parquet(resolved)

    def _write(
        self,
        df: pd.DataFrame,
        path: Path,
        mode: str = "overwrite",
        dedup_column: str | None = None,
    ) -> None:
        """写入 DataFrame 到 Parquet。"""
        self._storage.write_parquet(df, path, mode, dedup_column)

    def _register(
        self,
        key: str,
        path: Path,
        df: pd.DataFrame,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """在注册表中注册数据。"""
        if not path.is_absolute():
            resolved_path = self._storage.root.joinpath(path)
        else:
            resolved_path = path
        entry = self._storage.build_entry(resolved_path, df, extra)
        self._storage.set_entry(key, entry)

    def _get_meta(self, key: str) -> dict[str, Any] | None:
        """获取元数据条目。"""
        return self._storage.get_entry(key)

    def get_meta(self, key: str) -> dict[str, Any] | None:
        """通过键获取元数据条目。"""
        return self._get_meta(key)

    def list_cached_keys(self, prefix: str | None = None) -> list[str]:
        """列出注册表中的所有键。"""
        meta = self._storage.load_meta()
        if prefix:
            return [k for k in meta if k.startswith(prefix)]
        return list(meta.keys())
