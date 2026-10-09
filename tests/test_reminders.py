import threading
import time
from datetime import datetime, timedelta

from jarvis.skills.reminders import ReminderScheduler, ReminderStore


def test_store_roundtrip(tmp_path):
    store = ReminderStore(tmp_path / "r.db")
    due = datetime(2030, 1, 1, 9, 30)
    reminder = store.add("call mom", due)
    assert reminder.id == 1
    pending = store.pending()
    assert [(r.task, r.due, r.connector) for r in pending] == [("call mom", due, "to")]
    store.close()


def test_reminders_survive_restart(tmp_path):
    ReminderStore(tmp_path / "r.db").add("water the plants", datetime(2030, 1, 1))
    assert [r.task for r in ReminderStore(tmp_path / "r.db").pending()] == ["water the plants"]


def test_due_and_delivery(tmp_path):
    store = ReminderStore(tmp_path / "r.db")
    now = datetime(2026, 10, 6, 14, 0)
    store.add("past", now - timedelta(minutes=1))
    store.add("future", now + timedelta(hours=1))
    assert [r.task for r in store.due(now)] == ["past"]
    store.mark_delivered(store.due(now)[0].id)
    assert store.due(now) == []
    assert [r.task for r in store.pending()] == ["future"]


def test_clear_pending(tmp_path):
    store = ReminderStore(tmp_path / "r.db")
    store.add("a", datetime(2030, 1, 1))
    store.add("b", datetime(2030, 1, 2))
    assert store.clear_pending() == 2
    assert store.pending() == []


def test_scheduler_delivers_once(tmp_path):
    store = ReminderStore(tmp_path / "r.db")
    store.add("stretch", datetime.now() - timedelta(seconds=1))
    delivered = []
    got_one = threading.Event()

    def on_due(reminder):
        delivered.append(reminder.task)
        got_one.set()

    scheduler = ReminderScheduler(store, on_due, interval=0.05)
    scheduler.start()
    assert got_one.wait(2)
    time.sleep(0.2)  # several more polls: must not announce it again
    scheduler.stop()
    scheduler.join(1)
    assert delivered == ["stretch"]


def test_scheduler_survives_failing_callback(tmp_path):
    store = ReminderStore(tmp_path / "r.db")
    store.add("a", datetime.now() - timedelta(seconds=1))

    def boom(_):
        raise RuntimeError("speaker unplugged")

    assert ReminderScheduler(store, boom).check() == 1
    assert store.pending() == []
