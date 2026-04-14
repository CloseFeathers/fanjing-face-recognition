"""
VideoSource —— 本地视频文件采集源。

关键设计：
  - timestamp_ms 直接取自视频容器 (CAP_PROP_POS_MSEC)，
    同一文件重复运行时时间戳序列完全一致。
  - 默认不丢帧，按顺序逐帧输出（保证离线可复现）。
  - 提供 realtime 开关：
      realtime=False  尽可能快跑完（默认）
      realtime=True   按视频原始帧率限速播放
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

from .frame import Frame


class VideoSource:
    """视频文件帧源。

    Parameters:
        path:      视频文件路径
        realtime:  True = 按视频时间实时播放；False = 尽可能快跑完
    """

    def __init__(self, path: str, realtime: bool = False) -> None:
        self._path = path
        self._realtime = realtime
        self.source_id = f"file:{Path(path).name}"

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._video_fps: float = 0.0
        self._total_frames: int = 0

        # ---------- FPS 统计 ----------
        self._fps_window_start: float = 0.0
        self._fps_window_count: int = 0
        self._current_fps: float = 0.0

        # ---------- realtime 限速 ----------
        self._last_read_wall: float = 0.0
        self._last_video_ts: float = 0.0

    # ==================================================================
    # 生命周期
    # ==================================================================

    def open(self) -> "VideoSource":
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self._path}")
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
    # 读帧
    # ==================================================================

    def read(self) -> Optional[Frame]:
        """顺序读取下一帧。视频读完返回 None。"""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, img = self._cap.read()
        if not ret:
            return None

        # 视频容器时间戳（毫秒）—— 保证可复现
        ts_ms: float = self._cap.get(cv2.CAP_PROP_POS_MSEC)

        # realtime 限速
        if self._realtime and self._frame_id > 0:
            delta_video = ts_ms - self._last_video_ts          # 视频时间差
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
            dropped_frames=0,  # 视频文件模式不丢帧
        )
        self._frame_id += 1

        # 更新 FPS
        self._fps_window_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_window_count / elapsed
            self._fps_window_count = 0
            self._fps_window_start = now

        return frame

    # ==================================================================
    # 属性
    # ==================================================================

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def video_fps(self) -> float:
        """视频文件自身的帧率。"""
        return self._video_fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def dropped_frames(self) -> int:
        return 0  # 视频文件模式永不丢帧

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
