"""交易账本 — SQLite 持久化用于崩溃恢复。

每笔委托在发送给券商之前写入磁盘。
重启后 TradeLedger.scan_pending() 返回所有非终态订单用于对账。
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from app.config.paths import TRADE_LEDGER_FILE

if TYPE_CHECKING:
    from app.engine.live.gateway.base import OrderRequest

logger = logging.getLogger(__name__)

_TERMINAL = {"filled", "cancelled", "rejected"}


class TradeLedger:
    """基于 SQLite 的订单持久化，用于崩溃恢复。

    线程安全（通过 RLock）。所有时间戳存储为 ISO8601 字符串。
    """

    def __init__(self, db_path: Path | str = TRADE_LEDGER_FILE) -> None:
        self._db = Path(db_path)
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._init_db()
        logger.debug("TradeLedger initialized with db=%s", self._db)

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self._db)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    local_id       TEXT PRIMARY KEY,
                    broker_id     TEXT,
                    symbol        TEXT NOT NULL,
                    direction     TEXT NOT NULL,
                    quantity      INTEGER NOT NULL,
                    limit_price   INTEGER,
                    status        TEXT NOT NULL,
                    reject_reason TEXT,
                    filled_qty    INTEGER DEFAULT 0,
                    avg_price     INTEGER DEFAULT 0,
                    created_ts    TEXT NOT NULL,
                    updated_ts    TEXT NOT NULL,
                    broker_order_id TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON orders(status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_broker_id ON orders(broker_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id   TEXT PRIMARY KEY,
                    broker_id  TEXT NOT NULL,
                    symbol     TEXT NOT NULL,
                    direction  TEXT NOT NULL,
                    qty        INTEGER NOT NULL,
                    price_int  INTEGER NOT NULL,
                    traded_at  TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_broker ON trades(broker_id)"
            )
            for col, dtype in [
                ("filled_qty", "INTEGER DEFAULT 0"),
                ("avg_price", "INTEGER DEFAULT 0"),
            ]:
                with suppress(sqlite3.OperationalError):
                    conn.execute(f"ALTER TABLE orders ADD COLUMN {col} {dtype}")
            conn.commit()
            conn.close()

    def write_order(self, order: OrderRequest, broker_order_id: str = "") -> None:
        """写入一笔委托（在发给券商之前调用）。"""
        with self._lock:
            conn = sqlite3.connect(self._db)
            now = datetime.now().isoformat()
            conn.execute(
                """
                INSERT OR REPLACE INTO orders
                    (local_id, broker_id, symbol, direction, quantity, limit_price,
                     status, filled_qty, avg_price, created_ts, updated_ts,
                     broker_order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                """,
                (
                    order.order_id,
                    broker_order_id,
                    order.symbol,
                    order.direction.value,
                    order.quantity,
                    order.limit_price,
                    "submitted",
                    now,
                    now,
                    broker_order_id,
                ),
            )
            conn.commit()
            conn.close()

    def record_trade(
        self,
        broker_id: str,
        trade_id: str,
        symbol: str,
        direction: str,
        qty: int,
        price_int: int,
    ) -> bool:
        """写入一笔成交（去重: trade_id 唯一约束）。

        返回:
            True 如果记录成功，False 如果是重复
        """
        with self._lock:
            conn = sqlite3.connect(self._db)
            now = datetime.now().isoformat()
            try:
                conn.execute(
                    """
                    INSERT INTO trades
                        (trade_id, broker_id, symbol, direction, qty,
                         price_int, traded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (trade_id, broker_id, symbol, direction, qty, price_int, now),
                )
                conn.commit()
                inserted = True
            except sqlite3.IntegrityError:
                inserted = False
            finally:
                conn.close()
            return inserted

    def update_status(
        self,
        local_id: str,
        status: str,
        reject_reason: str = "",
    ) -> None:
        """更新订单状态。"""
        with self._lock:
            conn = sqlite3.connect(self._db)
            now = datetime.now().isoformat()
            query = (
                "UPDATE orders SET status=?, reject_reason=?, updated_ts=? "
                "WHERE local_id=?"
            )
            conn.execute(query, (status, reject_reason, now, local_id))
            conn.commit()
            conn.close()

    def scan_pending(self) -> list[dict[str, object]]:
        """扫描所有非终态订单，用于重启后恢复。

        返回:
            字典列表，包含 {local_id, broker_id, symbol, direction, quantity, status, created_ts}
        """
        with self._lock:
            conn = sqlite3.connect(self._db)
            sql = """
                SELECT local_id, broker_id, symbol, direction, quantity,
                       limit_price, status, created_ts, broker_order_id
                FROM orders
                WHERE status NOT IN (?, ?, ?)
                ORDER BY created_ts ASC
                """
            terminal_list = ["filled", "cancelled", "rejected"]
            cursor = conn.execute(sql, (*terminal_list,))
            rows = cursor.fetchall()
            conn.close()

        return [
            {
                "local_id": r[0],
                "broker_id": r[1],
                "symbol": r[2],
                "direction": r[3],
                "quantity": r[4],
                "limit_price": r[5],
                "status": r[6],
                "created_ts": r[7],
                "broker_order_id": r[8],
            }
            for r in rows
        ]

    def list_all(self, limit: int = 100) -> list[dict[str, object]]:
        """列出最近的订单历史。"""
        with self._lock:
            conn = sqlite3.connect(self._db)
            cursor = conn.execute(
                "SELECT * FROM orders ORDER BY created_ts DESC LIMIT ?",
                (limit,),
            )
            cols = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
            conn.close()
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def clear_filled(self, before_ts: datetime | None = None) -> int:
        """删除已完成的旧订单。

        返回:
            删除的行数
        """
        if before_ts is None:
            before_ts = datetime.now() - timedelta(days=30)
        with self._lock:
            conn = sqlite3.connect(self._db)
            sql = "DELETE FROM orders WHERE status IN (?, ?, ?) AND updated_ts < ?"
            params = ["filled", "cancelled", "rejected", before_ts.isoformat()]
            cursor = conn.execute(sql, params)
            conn.commit()
            deleted = cursor.rowcount
            conn.close()
        return deleted
