"""Multi-turn session persistence. SQLite here; DynamoDB in the AWS version.

The schema mirrors the DynamoDB single-table design so the swap is mechanical:

    DynamoDB:  PK = session_id, SK = turn (zero-padded), attrs = role, content, ...
    SQLite:    PRIMARY KEY (session_id, turn)

Why the server owns history (instead of the client resending it): the client
can't tamper with what the model "said", history survives refreshes/devices,
and you get an audit log of every answer + its citations for free. Why a
*store* at all: in a serverless deploy (Lambda) there is no process memory
between requests — any state must live outside the function.

The store is behind a tiny interface (append / history / exists), so a
DynamoDB implementation is ~30 lines of boto3 put_item/query.
"""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class SessionStore:
    def __init__(self, path: Path | str) -> None:
        # check_same_thread=False + a lock: FastAPI serves requests on a threadpool
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    turn       INTEGER NOT NULL,
                    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content    TEXT NOT NULL,
                    meta       TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, turn)
                );
                """
            )

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    def append(self, session_id: str, role: str, content: str, meta: dict | None = None) -> int:
        """Append a message; returns its turn number. Atomic per session."""
        with self.lock, self.conn:
            (turn,) = self.conn.execute(
                "SELECT COALESCE(MAX(turn), -1) + 1 FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
            self.conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn, role, content, json.dumps(meta or {}), time.time()),
            )
        return turn

    def history(self, session_id: str, last_n: int | None = None) -> list[dict]:
        """Messages oldest-first, optionally only the most recent last_n."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT turn, role, content, meta FROM messages WHERE session_id = ? ORDER BY turn",
                (session_id,),
            ).fetchall()
        if last_n is not None:
            rows = rows[-last_n:] if last_n else []
        return [{"turn": t, "role": r, "content": c, "meta": json.loads(m)} for t, r, c, m in rows]

    def exists(self, session_id: str) -> bool:
        with self.lock:
            row = self.conn.execute("SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (session_id,)).fetchone()
        return row is not None

    def delete(self, session_id: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
