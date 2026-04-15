"""
CameraSource — Real-time camera capture source.

Key design:
  - Independent background thread continuously grabs frames (producer)
  - Frame buffer queue size strictly 1, keeps only latest frame
  - Consumer gets frames via read(); waits if queue is empty
  - Uses time.monotonic() for timestamps, immune to system time jumps
  - Real-time FPS and cumulative dropped_frames statistics
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from .frame import Frame


class CameraSource:
    """Camera frame source (producer-consumer model).

    Parameters:
        device:     OpenCV device index, default 0
        api_pref:   Backend preference, cv2.CAP_DSHOW recommended on Windows
    """

    def __init__(self, device: int = 0, api_pref: int = cv2.CAP_ANY) -> None:
        self._device = device
        self._api_pref = api_pref
        self.source_id = f"camera:{device}"

        # ---------- State ----------
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # ---------- Frame buffer (size = 1) ----------
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._new_frame_event = threading.Event()

        # ---------- Counters ----------
        self._grabbed_count: int = 0      # Total frames grabbed by background thread
        self._consumed_count: int = 0     # Frames consumed by consumer
        self._dropped_frames: int = 0     # Cumulative dropped frames
        self._frame_id: int = 0           # External incrementing frame ID

        # ---------- FPS ----------
        self._fps_window_start: float = 0.0
        self._fps_window_count: int = 0
        self._current_fps: float = 0.0

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def open(self) -> "CameraSource":
        """Open camera and start background capture thread."""
        self._cap = cv2.VideoCapture(self._device, self._api_pref)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._device}")
        self._running = True
        self._fps_window_start = time.monotonic()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop capture thread and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # context-manager 支持
    def __enter__(self) -> "CameraSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ==================================================================
    # Background capture thread (producer)
    # ==================================================================

    def _capture_loop(self) -> None:
        while self._running:
            ret, img = self._cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            ts = time.monotonic() * 1000.0  # Convert to milliseconds

            with self._lock:
                if self._latest_frame is not None:
                    # Old frame not consumed, discard
                    self._dropped_frames += 1
                self._latest_frame = img
                self._latest_ts = ts
                self._grabbed_count += 1

            self._new_frame_event.set()

    # ==================================================================
    # Consumer interface
    # ==================================================================

    def read(self, timeout: float = 5.0) -> Optional[Frame]:
        """Block and wait for latest frame.

        Returns:
            Frame | None  Returns None if timeout or closed
        """
        if not self._running:
            return None

        if not self._new_frame_event.wait(timeout=timeout):
            return None

        with self._lock:
            img = self._latest_frame
            ts = self._latest_ts
            self._latest_frame = None  # Mark as consumed
            dropped = self._dropped_frames

        self._new_frame_event.clear()

        if img is None:
            return None

        h, w = img.shape[:2]

        frame = Frame(
            image=img,
            timestamp_ms=ts,
            frame_id=self._frame_id,
            source_id=self.source_id,
            width=w,
            height=h,
            dropped_frames=dropped,
        )
        self._frame_id += 1
        self._consumed_count += 1

        # Update FPS (sliding 1-second window)
        self._fps_window_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_window_count / elapsed
            self._fps_window_count = 0
            self._fps_window_start = now

        return frame

    # ==================================================================
    # Statistics
    # ==================================================================

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def is_open(self) -> bool:
        return self._running
