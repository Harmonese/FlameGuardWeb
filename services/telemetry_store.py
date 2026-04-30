from __future__ import annotations

from collections import deque
from threading import RLock
from typing import Iterable


class TelemetryStore:
    """Small in-memory ring buffer for dashboard telemetry."""

    def __init__(self, maxlen: int = 600) -> None:
        self._rows: deque[dict] = deque(maxlen=maxlen)
        self._lock = RLock()

    def append(self, row: dict) -> None:
        with self._lock:
            self._rows.append(dict(row))

    def latest(self) -> dict | None:
        with self._lock:
            if not self._rows:
                return None
            return dict(self._rows[-1])

    def tail(self, limit: int = 120) -> list[dict]:
        with self._lock:
            if limit <= 0:
                return []
            return [dict(row) for row in list(self._rows)[-limit:]]

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()

    def replace(self, rows: Iterable[dict]) -> None:
        with self._lock:
            self._rows.clear()
            for row in rows:
                self._rows.append(dict(row))
