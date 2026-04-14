"""
CameraSource —— 摄像头实时采集源。

关键设计：
  - 独立后台线程持续抓帧（生产者）
  - 帧缓冲队列大小严格为 1，只保留最新帧
  - 消费者通过 read() 取帧；若队列为空则等待
  - 使用 time.monotonic() 作为时间戳来源，免疫系统时间跳变
  - 实时统计 FPS 与累计 dropped_frames
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from .frame import Frame


class CameraSource:
    """摄像头帧源（生产者-消费者模型）。

    Parameters:
        device:     OpenCV 设备索引, 默认 0
        api_pref:   后端偏好, Windows 上推荐 cv2.CAP_DSHOW
    """

    def __init__(self, device: int = 0, api_pref: int = cv2.CAP_ANY) -> None:
        self._device = device
        self._api_pref = api_pref
        self.source_id = f"camera:{device}"

        # ---------- 状态 ----------
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # ---------- 帧缓冲（大小 = 1） ----------
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._new_frame_event = threading.Event()

        # ---------- 计数 ----------
        self._grabbed_count: int = 0      # 后台线程总抓帧数
        self._consumed_count: int = 0     # 消费者已取走帧数
        self._dropped_frames: int = 0     # 累计丢弃帧数
        self._frame_id: int = 0           # 对外递增帧号

        # ---------- FPS ----------
        self._fps_window_start: float = 0.0
        self._fps_window_count: int = 0
        self._current_fps: float = 0.0

    # ==================================================================
    # 生命周期
    # ==================================================================

    def open(self) -> "CameraSource":
        """打开摄像头并启动后台采集线程。"""
        self._cap = cv2.VideoCapture(self._device, self._api_pref)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self._device}")
        self._running = True
        self._fps_window_start = time.monotonic()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        """停止采集线程并释放资源。"""
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
    # 后台采集线程（生产者）
    # ==================================================================

    def _capture_loop(self) -> None:
        while self._running:
            ret, img = self._cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            ts = time.monotonic() * 1000.0  # 转毫秒

            with self._lock:
                if self._latest_frame is not None:
                    # 旧帧未被消费，丢弃
                    self._dropped_frames += 1
                self._latest_frame = img
                self._latest_ts = ts
                self._grabbed_count += 1

            self._new_frame_event.set()

    # ==================================================================
    # 消费者接口
    # ==================================================================

    def read(self, timeout: float = 5.0) -> Optional[Frame]:
        """阻塞等待并返回最新帧。

        Returns:
            Frame | None  如果超时或已关闭则返回 None
        """
        if not self._running:
            return None

        if not self._new_frame_event.wait(timeout=timeout):
            return None

        with self._lock:
            img = self._latest_frame
            ts = self._latest_ts
            self._latest_frame = None  # 标记已消费
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

        # 更新 FPS（滑动 1 秒窗口）
        self._fps_window_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_window_count / elapsed
            self._fps_window_count = 0
            self._fps_window_start = now

        return frame

    # ==================================================================
    # 统计信息
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
