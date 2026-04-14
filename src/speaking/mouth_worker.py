"""
MouthWorker — 异步说话状态检测线程。

Per-track latest-only: 每个 track 独立任务槽，新帧覆盖旧帧。
主线程 submit face crop → worker 线程分析 → 结果回写到共享 dict。
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np

from .mouth_analyzer import MouthAnalyzer

logger = logging.getLogger(__name__)


class MouthWorker:
    """异步说话状态检测 worker。"""

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
        """Per-track latest-only 提交。"""
        with self._lock:
            self._pending[track_id] = (face_crop, timestamp_ms)
        self._event.set()

    def get_results(self) -> Dict[int, str]:
        """读取最新的 track_id -> status 映射。"""
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
