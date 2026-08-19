"""Relational SQLite database store for conversations and messages."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List


class SQLiteStore:
    """Relational SQLite database store for conversations and messages."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent.parent / "data" / "sessions.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    structured_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            conn.commit()

    def create_conversation(self, title: str = "New Clinical Session") -> str:
        conv_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (conv_id, title),
            )
            conn.commit()
        return conv_id

    def list_conversations(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        structured_response: Dict[str, Any] | None = None,
    ) -> str:
        msg_id = str(uuid.uuid4())
        resp_json = json.dumps(structured_response) if structured_response else None
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO messages (id, conversation_id, role, content, structured_response)
                   VALUES (?, ?, ?, ?, ?)""",
                (msg_id, conversation_id, role, content, resp_json),
            )
            conn.commit()
        return msg_id

    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT role, content, structured_response FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
            result = []
            for row in rows:
                r = dict(row)
                if r.get("structured_response"):
                    try:
                        r["structured_response"] = json.loads(r["structured_response"])
                    except Exception:
                        pass
                result.append(r)
            return result
