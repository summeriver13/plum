"""分层约束：公开层不得 import 私有策略层。

私有层（graft、production、signals/kapok、strategy/migrated）在开源仓里不存在，
公开层一旦 import 它们，公开仓就无法独立运行。
"""

from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent / "app"

PRIVATE_PREFIXES = (
    "graft/",
    "production/",
    "signals/kapok/",
    "strategy/migrated/",
)

FORBIDDEN_IMPORTS = ("app.graft", "app.production", "app.signals.kapok")


def _public_files() -> list[Path]:
    files: list[Path] = []
    for path in APP_ROOT.rglob("*.py"):
        rel = path.relative_to(APP_ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in PRIVATE_PREFIXES):
            continue
        files.append(path)
    return sorted(files)


@pytest.mark.parametrize("path", _public_files(), ids=lambda path: path.name)
def test_public_layer_does_not_import_private_layer(path: Path) -> None:
    rel = path.relative_to(APP_ROOT).as_posix()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith(("from ", "import ")):
            continue
        for forbidden in FORBIDDEN_IMPORTS:
            assert forbidden not in stripped, (
                f"{rel}:{lineno} 公开层 import 了私有层 {forbidden}"
            )
