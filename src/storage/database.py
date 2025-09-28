"""Lightweight ORM-like layer backed by sqlite3."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from src.config import get_settings

from .models import Swipe, SwipeEvent, User


_CONNECTION: Optional[sqlite3.Connection] = None
_LOCK = threading.Lock()


def _ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _database_path() -> str:
    uri = get_settings().database_uri
    if uri.startswith("sqlite:///"):
        location = uri[len("sqlite:///") :]
        if location == ":memory:":
            return ":memory:"
        db_path = Path(location)
        if db_path != Path(":memory:"):
            _ensure_parent_directory(db_path)
        return str(db_path)
    raise RuntimeError(
        "This lightweight storage layer currently supports only sqlite URIs."
    )


def _connect() -> sqlite3.Connection:
    global _CONNECTION
    if _CONNECTION is None:
        path = _database_path()
        _CONNECTION = sqlite3.connect(
            path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        _CONNECTION.row_factory = sqlite3.Row
        _CONNECTION.execute("PRAGMA foreign_keys = ON")
    return _CONNECTION


def init_db() -> None:
    """Create database tables if they do not exist."""

    with _LOCK:
        conn = _connect()
        cursor = conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                title TEXT,
                company TEXT,
                bio TEXT,
                preferences TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS swipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(target_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                matched_user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, matched_user_id),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(matched_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS match_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL UNIQUE,
                score REAL NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(match_id) REFERENCES matches(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS swipe_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                swipe_id INTEGER,
                event_type TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                score REAL,
                payload TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(swipe_id) REFERENCES swipes(id) ON DELETE SET NULL
            );
            """
        )
        cursor.close()


def reset_database() -> None:
    """Drop and recreate all tables (mainly for tests)."""

    with _LOCK:
        conn = _connect()
        cursor = conn.cursor()
        cursor.executescript(
            """
            DROP TABLE IF EXISTS swipe_events;
            DROP TABLE IF EXISTS match_scores;
            DROP TABLE IF EXISTS matches;
            DROP TABLE IF EXISTS swipes;
            DROP TABLE IF EXISTS users;
            """
        )
        cursor.close()
    init_db()


def _serialize_datetime(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat()


def _now_text() -> str:
    return _serialize_datetime(datetime.now(UTC))


@contextmanager
def transaction_scope(commit: bool = True) -> Iterator[sqlite3.Cursor]:
    """Provide a transactional cursor with simple retry logic."""

    settings = get_settings()
    attempts = max(1, settings.max_retries)
    delay = settings.retry_backoff

    for attempt in range(attempts):
        with _LOCK:
            conn = _connect()
            cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
            else:
                conn.rollback()
            cursor.close()
            break
        except sqlite3.OperationalError:
            cursor.close()
            conn.rollback()
            if attempt >= attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except Exception:
            cursor.close()
            conn.rollback()
            raise


def create_user(user: User) -> User:
    with transaction_scope() as cursor:
        cursor.execute(
            """
            INSERT INTO users (email, full_name, title, company, bio, preferences, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user.email,
                user.full_name,
                user.title,
                user.company,
                user.bio,
                json.dumps(user.preferences),
                int(user.is_active),
                _serialize_datetime(user.created_at),
                _serialize_datetime(user.updated_at),
            ),
        )
        user.id = cursor.lastrowid
    return user


def get_user(user_id: int) -> Optional[User]:
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    return User(
        id=row["id"],
        email=row["email"],
        full_name=row["full_name"],
        title=row["title"],
        company=row["company"],
        bio=row["bio"],
        preferences=json.loads(row["preferences"] or "[]"),
        is_active=bool(row["is_active"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def create_swipe(swipe: Swipe) -> Swipe:
    with transaction_scope() as cursor:
        cursor.execute(
            """
            INSERT INTO swipes (user_id, target_id, action, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                swipe.user_id,
                swipe.target_id,
                swipe.action,
                _serialize_datetime(swipe.created_at),
            ),
        )
        swipe.id = cursor.lastrowid
    return swipe


def has_mutual_like(user_id: int, target_id: int) -> bool:
    conn = _connect()
    row = conn.execute(
        """
        SELECT id FROM swipes
        WHERE user_id = ? AND target_id = ? AND action = 'like'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (target_id, user_id),
    ).fetchone()
    return row is not None


def create_matches(
    user_id: int,
    target_id: int,
    score: float,
    common_interests: Iterable[str],
) -> List[int]:
    details_json = json.dumps({"common_interests": list(common_interests)})
    created_at = _now_text()
    match_ids: List[int] = []

    with transaction_scope() as cursor:
        for left, right in ((user_id, target_id), (target_id, user_id)):
            cursor.execute(
                """
                INSERT OR IGNORE INTO matches (user_id, matched_user_id, created_at)
                VALUES (?, ?, ?)
                """,
                (left, right, created_at),
            )
            if cursor.rowcount == 0:
                existing = cursor.execute(
                    "SELECT id FROM matches WHERE user_id = ? AND matched_user_id = ?",
                    (left, right),
                ).fetchone()
                match_id = existing["id"]
            else:
                match_id = cursor.lastrowid
            cursor.execute(
                """
                INSERT OR REPLACE INTO match_scores (match_id, score, details, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (match_id, score, details_json, created_at),
            )
            match_ids.append(match_id)
    return match_ids


def log_swipe_event(event: SwipeEvent) -> SwipeEvent:
    with transaction_scope() as cursor:
        cursor.execute(
            """
            INSERT INTO swipe_events (swipe_id, event_type, user_id, target_id, action, score, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.swipe_id,
                event.event_type,
                event.user_id,
                event.target_id,
                event.action,
                event.score,
                json.dumps(event.payload),
                _serialize_datetime(event.created_at),
            ),
        )
        event.id = cursor.lastrowid
    return event


def fetch_matches_for_user(user_id: int) -> List[Dict[str, object]]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT m.id as match_id, m.created_at, u.id as partner_id, u.full_name, u.title, u.company,
               u.preferences, s.score, s.details
        FROM matches m
        LEFT JOIN users u ON u.id = m.matched_user_id
        LEFT JOIN match_scores s ON s.match_id = m.id
        WHERE m.user_id = ?
        ORDER BY m.created_at DESC
        """,
        (user_id,),
    ).fetchall()

    results: List[Dict[str, object]] = []
    for row in rows:
        preferences = json.loads(row["preferences"] or "[]") if row["preferences"] else []
        details = json.loads(row["details"] or "{}") if row["details"] else {}
        results.append(
            {
                "id": row["match_id"],
                "user_id": row["partner_id"],
                "name": row["full_name"],
                "title": row["title"],
                "company": row["company"],
                "match_score": row["score"],
                "preferences": preferences,
                "common_interests": details.get("common_interests", []),
                "created_at": row["created_at"],
            }
        )
    return results


def count_rows(table: str) -> int:
    conn = _connect()
    row = conn.execute(f"SELECT COUNT(*) as count FROM {table}").fetchone()
    return int(row["count"])


def fetch_swipe_events() -> List[Dict[str, object]]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT id, swipe_id, event_type, user_id, target_id, action, score, payload, created_at
        FROM swipe_events
        ORDER BY id
        """
    ).fetchall()
    events: List[Dict[str, object]] = []
    for row in rows:
        events.append(
            {
                "id": row["id"],
                "swipe_id": row["swipe_id"],
                "event_type": row["event_type"],
                "user_id": row["user_id"],
                "target_id": row["target_id"],
                "action": row["action"],
                "score": row["score"],
                "payload": json.loads(row["payload"] or "{}"),
                "created_at": row["created_at"],
            }
        )
    return events

