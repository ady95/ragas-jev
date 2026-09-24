"""SQLite key-value cache for judge answers.

Keys are hashes of (model, masked state, question payload); values are the
numeric answers only, so the cache file never holds evaluated text.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def stable_hash(*parts: Any) -> str:
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AnswerCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT value FROM answers WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO answers (key, value) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
