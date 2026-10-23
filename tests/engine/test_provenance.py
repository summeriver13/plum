"""运行溯源（provenance）单元测试。

覆盖指纹语义：同一份副本稳定、增删文件变化、指数桶能捕捉「同大小改写」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.engine.backtest import provenance as provenance_mod
from app.engine.backtest.provenance import (
    code_state,
    collect_provenance,
    market_fingerprint,
    repo_root,
)


def _make_copy(root: Path, files: dict[str, bytes]) -> Path:
    """在 root 下造出给定文件（相对路径 → 内容，越界路径直接拒绝）。"""
    base = root.resolve()
    for rel, content in files.items():
        path = (base / rel).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"越界路径: {rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


class TestMarketFingerprint:
    def test_missing_dir(self, tmp_path: Path) -> None:
        info = market_fingerprint(tmp_path / "nope")
        assert info["available"] is False

    def test_counts_files_and_digest_is_stable(self, tmp_path: Path) -> None:
        copy = _make_copy(
            tmp_path / "1d",
            {"SH/600000.parquet": b"aaaa", "SZ/000001.parquet": b"bb"},
        )
        first = market_fingerprint(copy)
        second = market_fingerprint(copy)
        assert first["available"] is True
        assert first["files"] == 2
        assert first["total_bytes"] == 6
        assert first["tree_digest"] == second["tree_digest"]

    def test_new_file_changes_digest(self, tmp_path: Path) -> None:
        copy = _make_copy(tmp_path / "1d", {"SH/600000.parquet": b"aa"})
        before = market_fingerprint(copy)["tree_digest"]
        _make_copy(copy, {"SH/600001.parquet": b"cc"})
        assert market_fingerprint(copy)["tree_digest"] != before

    def test_index_bucket_catches_same_size_rewrite(self, tmp_path: Path) -> None:
        """文件数与大小不变、内容被改写：树摘要不变，指数桶内容摘要必须变化。"""
        copy = _make_copy(
            tmp_path / "1d",
            {"SH/600000.parquet": b"aa", "IDX/399303.parquet": b"1111"},
        )
        before = market_fingerprint(copy)
        _make_copy(copy, {"IDX/399303.parquet": b"2222"})
        after = market_fingerprint(copy)
        assert before["index_digest"] is not None
        assert after["tree_digest"] == before["tree_digest"]
        assert after["index_digest"] != before["index_digest"]


class TestCodeState:
    def test_repo_state(self) -> None:
        state = code_state(repo_root())
        assert state["available"] is True
        assert len(str(state["commit"])) >= 7
        assert isinstance(state["dirty"], bool)

    def test_git_unavailable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise OSError("git 不存在")

        monkeypatch.setattr(provenance_mod.subprocess, "run", _boom)
        assert code_state(tmp_path)["available"] is False


class TestCollectProvenance:
    def test_shape(self) -> None:
        info = collect_provenance()
        assert "market" in info
        assert "code" in info
        assert str(info["market"]["path"]).endswith("1d")
