"""
MouthWorker — Async speaking state detection thread.

Per-track latest-only: Each track has independent task slot, new frame overwrites old frame.
Main thread submit face crop → worker thread analyzes → results write back to shared dict.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np

from .mouth_analyzer import MouthAnalyzer

logger = logging.getLogger(__name__)


class MouthWorker:
    """Async speaking state detection worker."""

    def __init__(self, analyzer: MouthAnalyzer):
        self._analyzer = analyzer
        self._pending: Dict[int, Tuple[np.ndarray, float]] = {}
        self._results: Dict[int, str] = {}
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._pending.clear()
        self._results.clear()
        self._event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mouth-worker")
        self._thread.start()

    def stop(self):
        self._running = False
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def submit(self, track_id: int, face_crop: np.ndarray, timestamp_ms: float):
        """Per-track latest-only submission."""
        with self._lock:
            self._pending[track_id] = (face_crop, timestamp_ms)
        self._event.set()

    def get_results(self) -> Dict[int, str]:
        """Read latest track_id -> status mapping."""
        with self._lock:
            return dict(self._results)

    def _loop(self):
        while self._running:
            self._event.wait(timeout=1.0)
            self._event.clear()

            with self._lock:
                jobs = dict(self._pending)
                self._pending.clear()

            if not jobs:
                continue

            for track_id, (face_crop, ts_ms) in jobs.items():
                try:
                    result = self._analyzer.analyze(track_id, face_crop, ts_ms)
                    with self._lock:
                        self._results[track_id] = result.status
                except Exception as e:
                    logger.exception("[MouthWorker] track %s error: %s", track_id, e)
