"""Reminders: a SQLite store plus a background scheduler thread (roadmap step 5).

Reminders are written to SQLite, so they survive restarts: anything that came
due while Jarvis was off is announced as soon as it starts again.

Design note: the ``schedule`` library is built for *recurring* jobs ("every
day at 10:30") and keeps jobs in memory, so one-off reminders would vanish on
restart. A small thread that polls the database once a second is simpler and
persistent.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task        TEXT    NOT NULL,
    connector   TEXT    NOT NULL DEFAULT 'to',
    due_ts      REAL    NOT NULL,
    created_ts  REAL    NOT NULL,
    delivered   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders (delivered, due_ts);
"""


@dataclass
class Reminder:
    id: int
    task: str
    due: datetime
    created: datetime
    connector: str = "to"
    delivered: bool = False


class ReminderStore:
    """Thread-safe SQLite persistence for reminders."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    @staticmethod
    def _row(row: sqlite3.Row) -> Reminder:
        return Reminder(id=row["id"], task=row["task"], due=datetime.fromtimestamp(row["due_ts"]),
                        created=datetime.fromtimestamp(row["created_ts"]), connector=row["connector"],
                        delivered=bool(row["delivered"]))

    def add(self, task: str, due: datetime, connector: str = "to") -> Reminder:
        now = datetime.now()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO reminders (task, connector, due_ts, created_ts) VALUES (?, ?, ?, ?)",
                (task, connector, due.timestamp(), now.timestamp()))
        return Reminder(cursor.lastrowid, task, due, now, connector)

    def pending(self) -> List[Reminder]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE delivered = 0 ORDER BY due_ts").fetchall()
        return [self._row(r) for r in rows]

    def due(self, now: Optional[datetime] = None) -> List[Reminder]:
        now = now or datetime.now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE delivered = 0 AND due_ts <= ? ORDER BY due_ts",
                (now.timestamp(),)).fetchall()
        return [self._row(r) for r in rows]

    def mark_delivered(self, reminder_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE reminders SET delivered = 1 WHERE id = ?", (reminder_id,))

    def clear_pending(self) -> int:
        with self._lock, self._conn:
            return self._conn.execute("DELETE FROM reminders WHERE delivered = 0").rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class ReminderScheduler(threading.Thread):
    """Background thread: every ``interval`` seconds, deliver reminders that are due."""

    def __init__(self, store: ReminderStore, on_due: Callable[[Reminder], None], interval: float = 1.0):
        super().__init__(name="reminder-scheduler", daemon=True)
        self.store = store
        self.on_due = on_due
        self.interval = interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.check()
            self._stop_event.wait(self.interval)

    def check(self, now: Optional[datetime] = None) -> int:
        delivered = 0
        try:
            for reminder in self.store.due(now):
                self.store.mark_delivered(reminder.id)  # mark first: never announce twice
                try:
                    self.on_due(reminder)
                except Exception:  # a failing callback must not kill the scheduler
                    log.exception("Reminder callback failed")
                delivered += 1
        except sqlite3.Error:
            log.exception("Could not read reminders")
        return delivered

    def stop(self) -> None:
        self._stop_event.set()
