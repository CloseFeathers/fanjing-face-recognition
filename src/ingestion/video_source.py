"""
VideoSource — Local video file capture source.

Key design:
  - timestamp_ms directly from video container (CAP_PROP_POS_MSEC),
    timestamp sequence is identical when running same file repeatedly.
  - No frame dropping by default, outputs frames sequentially (ensures offline reproducibility).
  - Provides realtime switch:
      realtime=False  Run as fast as possible (default)
      realtime=True   Rate-limit playback to original video framerate
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

from .frame import Frame


class VideoSource:
    """Video file frame source.

    Parameters:
        path:      Video file path
        realtime:  True = real-time playback at video time; False = run as fast as possible
    """

    def __init__(self, path: str, realtime: bool = False) -> None:
        self._path = path
        self._realtime = realtime
        self.source_id = f"file:{Path(path).name}"

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._video_fps: float = 0.0
        self._total_frames: int = 0

        # ---------- FPS statistics ----------
        self._fps_window_start: float = 0.0
        self._fps_window_count: int = 0
        self._current_fps: float = 0.0

        # ---------- Realtime rate limiting ----------
        self._last_read_wall: float = 0.0
        self._last_video_ts: float = 0.0

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def open(self) -> "VideoSource":
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self._path}")
        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps_window_start = time.monotonic()
        self._last_read_wall = time.monotonic()
        self._last_video_ts = 0.0
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ==================================================================
    # Read frame
    # ==================================================================

    def read(self) -> Optional[Frame]:
        """Read next frame sequentially. Returns None when video ends."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, img = self._cap.read()
        if not ret:
            return None

        # Video container timestamp (milliseconds) — ensures reproducibility
        ts_ms: float = self._cap.get(cv2.CAP_PROP_POS_MSEC)

        # Realtime rate limiting
        if self._realtime and self._frame_id > 0:
            delta_video = ts_ms - self._last_video_ts          # Video time delta
            delta_wall = (time.monotonic() - self._last_read_wall) * 1000.0
            sleep_ms = delta_video - delta_wall
            if sleep_ms > 1.0:
                time.sleep(sleep_ms / 1000.0)

        self._last_read_wall = time.monotonic()
        self._last_video_ts = ts_ms

        h, w = img.shape[:2]
        frame = Frame(
            image=img,
            timestamp_ms=ts_ms,
            frame_id=self._frame_id,
            source_id=self.source_id,
            width=w,
            height=h,
            dropped_frames=0,  # Video file mode never drops frames
        )
        self._frame_id += 1

        # Update FPS
        self._fps_window_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_window_count / elapsed
            self._fps_window_count = 0
            self._fps_window_start = now

        return frame

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def video_fps(self) -> float:
        """The video file's native framerate."""
        return self._video_fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def dropped_frames(self) -> int:
        return 0  # Video file mode never drops frames

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
