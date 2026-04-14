"""
说话状态检测 (Phase 1) — 基于 SCRFD 5 点关键点的时序分析。

利用嘴角间距的时序变化推断说话状态:
- 说话时嘴角有节奏性运动 → motion_var 高
- 不说话时嘴角稳定 → motion_var 低
- 检测不可靠时 → occluded

输出三态: speaking / not_speaking / occluded
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class TrackMouthState:
    status: str          # "speaking" / "not_speaking" / "occluded" / "unknown"
    speaking_prob: float  # 0.0 - 1.0
    motion_var: float
    buffer_size: int
    timestamp_ms: float = 0.0


class MouthTracker:
    """Per-track 嘴部运动时序追踪器。"""

    def __init__(
        self,
        buffer_size: int = 15,
        speaking_threshold: float = 0.0003,
        min_buffer_frames: int = 8,
        min_det_score: float = 0.5,
        min_eye_dist_px: float = 15.0,
    ):
        self.buffer_size = buffer_size
        self.speaking_threshold = speaking_threshold
        self.min_buffer_frames = min_buffer_frames
        self.min_det_score = min_det_score
        self.min_eye_dist_px = min_eye_dist_px
        self._buffers: Dict[int, deque] = {}

    def update(self, track_id: int, kps5, det_score: float,
               bbox_xyxy, timestamp_ms: float = 0.0) -> TrackMouthState:
        """更新 track 的嘴部状态并返回判定结果。"""

        if kps5 is None or len(kps5) < 5 or det_score is None:
            return TrackMouthState("occluded", 0.0, 0.0, 0, timestamp_ms)

        pts = np.array(kps5, dtype=np.float32)
        left_eye, right_eye = pts[0], pts[1]
        left_mouth, right_mouth = pts[3], pts[4]

        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        if eye_dist < self.min_eye_dist_px:
            return TrackMouthState("occluded", 0.0, 0.0, 0, timestamp_ms)

        if det_score < self.min_det_score:
            return TrackMouthState("occluded", 0.0, 0.0, 0, timestamp_ms)

        x1, y1, x2, y2 = bbox_xyxy[:4]
        margin = max(x2 - x1, y2 - y1) * 0.1
        for px, py in [left_mouth, right_mouth]:
            if px < x1 - margin or px > x2 + margin or py < y1 - margin or py > y2 + margin:
                return TrackMouthState("occluded", 0.0, 0.0, 0, timestamp_ms)

        mouth_width = float(np.linalg.norm(right_mouth - left_mouth))
        mouth_ratio = mouth_width / eye_dist

        buf = self._buffers.get(track_id)
        if buf is None:
            buf = deque(maxlen=self.buffer_size)
            self._buffers[track_id] = buf
        buf.append(mouth_ratio)

        n = len(buf)
        if n < self.min_buffer_frames:
            return TrackMouthState("unknown", 0.0, 0.0, n, timestamp_ms)

        ratios = list(buf)
        diffs = [ratios[i+1] - ratios[i] for i in range(len(ratios) - 1)]
        motion_var = float(np.var(diffs)) if diffs else 0.0

        speaking_prob = min(motion_var / (self.speaking_threshold * 3), 1.0)

        if motion_var >= self.speaking_threshold:
            status = "speaking"
        else:
            status = "not_speaking"

        return TrackMouthState(status, speaking_prob, motion_var, n, timestamp_ms)

    def remove_track(self, track_id: int):
        self._buffers.pop(track_id, None)

    def reset(self):
        self._buffers.clear()
