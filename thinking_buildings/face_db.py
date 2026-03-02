from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("thinking_buildings")


class FaceDB:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._cache: Dict[str, List[np.ndarray]] = {}
        self._load_cache()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                source TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            );
        """)
        self._conn.commit()

    def _load_cache(self) -> None:
        self._cache.clear()
        rows = self._conn.execute("""
            SELECT p.name, e.embedding
            FROM embeddings e JOIN persons p ON e.person_id = p.id
        """).fetchall()
        for name, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            self._cache.setdefault(name, []).append(emb)
        logger.info("Loaded %d embeddings for %d persons", len(rows), len(self._cache))

    def add_person(self, name: str) -> int:
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO persons (name) VALUES (?)", (name,)
        )
        self._conn.commit()
        if cursor.lastrowid and cursor.rowcount > 0:
            return cursor.lastrowid
        row = self._conn.execute(
            "SELECT id FROM persons WHERE name = ?", (name,)
        ).fetchone()
        return row[0]

    def add_embedding(self, name: str, embedding: np.ndarray, source: Optional[str] = None) -> None:
        person_id = self.add_person(name)
        blob = embedding.astype(np.float32).tobytes()
        self._conn.execute(
            "INSERT INTO embeddings (person_id, embedding, source) VALUES (?, ?, ?)",
            (person_id, blob, source),
        )
        self._conn.commit()
        self._cache.setdefault(name, []).append(embedding.astype(np.float32))

    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        return self._cache

    def remove_person(self, name: str) -> bool:
        self._conn.execute("PRAGMA foreign_keys = ON")
        cursor = self._conn.execute("DELETE FROM persons WHERE name = ?", (name,))
        self._conn.commit()
        self._cache.pop(name, None)
        return cursor.rowcount > 0

    def list_persons(self) -> List[Tuple[str, int]]:
        rows = self._conn.execute("""
            SELECT p.name, COUNT(e.id)
            FROM persons p LEFT JOIN embeddings e ON p.id = e.person_id
            GROUP BY p.name ORDER BY p.name
        """).fetchall()
        return [(name, count) for name, count in rows]

    def close(self) -> None:
        self._conn.close()
