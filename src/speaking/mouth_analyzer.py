"""
MouthAnalyzer — 四层遮挡判断 + 时序 Speaking Expert。

四层架构:
  L1: Observability gate (ROI 太小/太糊/太暗/mesh 失败)
  L2: Self-occlusion hard gate (|yaw| > max_yaw)
  L3: Contour support (CLAHE + 法线梯度 → visible_ratio / max_gap)
  L4: Temporal fusion (连续 N 帧遮挡证据 → 确认 occluded)

Speaking Expert:
  多维口型时序分析 + 滞后状态机 (防闪烁)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from .mesh_detector import (
    INNER_LIP_INDICES,
    OUTER_LIP_INDICES,
    MeshDetector,
)

# ======================================================================
# 输出数据
# ======================================================================

@dataclass
class MouthState:
    status: str              # speaking / not_speaking / occluded / self_occluded / unobservable / unknown
    speaking_prob: float     # 0-1
    occlusion_prob: float    # 0-1
    confidence: float        # 整体置信度
    reason_code: str
    timestamp_ms: float = 0.0


# ======================================================================
# Per-track 内部数据
# ======================================================================

class _RunningStats:
    """增量均值/方差计算器。"""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        return (self.m2 / self.n) ** 0.5 if self.n > 1 else 0.0


class _TrackMouthState:
    """Per-track 状态机 + 环形缓存 + 基线。"""
    def __init__(self, buffer_size: int = 20):
        self.status = "unknown"
        self.speaking_prob_smoothed = 0.0

        self.occluded_streak = 0
        self.observable_streak = 0
        self.speaking_streak = 0
        self.not_speaking_streak = 0

        self.open_ratios: deque = deque(maxlen=buffer_size)
        self.width_ratios: deque = deque(maxlen=buffer_size)
        self.contour_supports: deque = deque(maxlen=buffer_size)
        self.timestamps: deque = deque(maxlen=buffer_size)

        self.baselines: Dict[str, _RunningStats] = {}


# ======================================================================
# MouthAnalyzer
# ======================================================================

class MouthAnalyzer:

    def __init__(
        self,
        mesh_detector: MeshDetector,
        buffer_size: int = 20,
        # L1
        min_crop_size: float = 40.0,
        min_blur_score: float = 10.0,
        # L2
        max_yaw: float = 60.0,
        # L3 contour support (法线两侧对比度)
        gradient_threshold: float = 18.0,
        min_visible_ratio: float = 0.45,
        max_missing_arc: int = 8,
        # L4
        occluded_confirm_frames: int = 4,
        observable_confirm_frames: int = 3,
        # Speaking
        speaking_threshold: float = 0.45,
        not_speaking_threshold: float = 0.25,
        speaking_confirm_frames: int = 5,
        not_speaking_confirm_frames: int = 8,
        min_speaking_buffer: int = 10,
    ):
        self._mesh = mesh_detector
        self._buffer_size = buffer_size
        self._min_crop_size = min_crop_size
        self._min_blur_score = min_blur_score
        self._max_yaw = max_yaw
        self._gradient_threshold = gradient_threshold
        self._min_visible_ratio = min_visible_ratio
        self._max_missing_arc = max_missing_arc
        self._occluded_confirm = occluded_confirm_frames
        self._observable_confirm = observable_confirm_frames
        self._spk_thresh = speaking_threshold
        self._nspk_thresh = not_speaking_threshold
        self._spk_confirm = speaking_confirm_frames
        self._nspk_confirm = not_speaking_confirm_frames
        self._min_spk_buf = min_speaking_buffer

        self._tracks: Dict[int, _TrackMouthState] = {}
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def analyze(
        self,
        track_id: int,
        face_crop_bgr: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> MouthState:
        ts = self._get_track(track_id)
        mesh = self._mesh.detect(face_crop_bgr)

        # ---- L1: Observability ----
        if mesh.landmarks_478 is None or mesh.mesh_confidence < 0.5:
            return self._emit(ts, "unobservable", 0.0, 1.0, "mesh_failed", timestamp_ms)
        if mesh.face_crop_size < self._min_crop_size:
            return self._emit(ts, "unobservable", 0.0, 1.0, "crop_too_small", timestamp_ms)
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < self._min_blur_score:
            return self._emit(ts, "unobservable", 0.0, 1.0, "too_blurry", timestamp_ms)

        # ---- L2: Self-occlusion ----
        if abs(mesh.head_yaw) > self._max_yaw:
            return self._emit(ts, "self_occluded", 0.0, 1.0,
                              f"yaw={mesh.head_yaw:.0f}", timestamp_ms)

        # ---- L3: Contour support ----
        h, w = face_crop_bgr.shape[:2]
        visible_ratio, max_gap = self._compute_contour_support(
            gray, mesh.landmarks_478, w, h
        )

        pose_bucket = self._get_pose_bucket(mesh.head_yaw)
        baseline = ts.baselines.get(pose_bucket)
        baseline_deviation = 0.0

        if visible_ratio >= self._min_visible_ratio and max_gap < self._max_missing_arc:
            is_occluded_frame = False
            if baseline is not None and baseline.n >= 5:
                baseline.update(visible_ratio)
            else:
                if baseline is None:
                    ts.baselines[pose_bucket] = _RunningStats()
                ts.baselines[pose_bucket].update(visible_ratio)
        else:
            is_occluded_frame = True
            if baseline is not None and baseline.n >= 5:
                baseline_deviation = max(0.0, baseline.mean - visible_ratio)

        ts.contour_supports.append(visible_ratio)

        # ---- L4: Temporal fusion ----
        if is_occluded_frame:
            ts.occluded_streak += 1
            ts.observable_streak = 0
        else:
            ts.observable_streak += 1
            ts.occluded_streak = 0

        if ts.occluded_streak >= self._occluded_confirm:
            reason = f"contour_vis={visible_ratio:.2f},gap={max_gap}"
            if baseline_deviation > 0:
                reason += f",baseline_dev={baseline_deviation:.2f}"
            return self._emit(ts, "occluded", 0.0, 0.8, reason, timestamp_ms)

        # ---- Speaking expert (only if observable) ----
        ts.open_ratios.append(mesh.mouth_open_ratio)
        ts.width_ratios.append(mesh.mouth_width_ratio)
        ts.timestamps.append(timestamp_ms)

        if len(ts.open_ratios) < self._min_spk_buf:
            return self._emit(ts, "unknown", 0.0, 0.0,
                              f"buffer={len(ts.open_ratios)}", timestamp_ms)

        speaking_prob = self._compute_speaking_prob(ts)

        alpha = 0.3
        ts.speaking_prob_smoothed = ts.speaking_prob_smoothed * (1 - alpha) + speaking_prob * alpha
        sp = ts.speaking_prob_smoothed

        # 滞后状态机
        if sp >= self._spk_thresh:
            ts.speaking_streak += 1
            ts.not_speaking_streak = 0
        elif sp < self._nspk_thresh:
            ts.not_speaking_streak += 1
            ts.speaking_streak = 0
        else:
            ts.speaking_streak = max(0, ts.speaking_streak - 1)
            ts.not_speaking_streak = max(0, ts.not_speaking_streak - 1)

        if ts.speaking_streak >= self._spk_confirm:
            status = "speaking"
        elif ts.not_speaking_streak >= self._nspk_confirm:
            status = "not_speaking"
        else:
            status = ts.status if ts.status in ("speaking", "not_speaking") else "not_speaking"

        return self._emit(ts, status, sp, 0.7,
                          f"prob={sp:.2f},raw={speaking_prob:.2f},vis={visible_ratio:.2f}",
                          timestamp_ms)

    # ------------------------------------------------------------------
    # L3: Contour support
    # ------------------------------------------------------------------

    def _compute_contour_support(
        self, gray: np.ndarray, pts: np.ndarray, w: int, h: int,
    ) -> Tuple[float, int]:
        """计算嘴部轮廓点的边界支持度。

        对每个轮廓点:
          1. 估算法线方向 (垂直于相邻两点连线)
          2. 沿法线方向采样内外两侧亮度, 计算对比度
          3. 对比度高 → 真正的唇缘边界 → supported
          4. 对比度低 → 被遮挡物覆盖 → not supported
        """
        clahe_gray = self._clahe.apply(gray)

        def _sample_avg(img, cx, cy, dx, dy, steps, img_w, img_h):
            total, cnt = 0.0, 0
            for s in range(1, steps + 1):
                sx = int(round(cx + dx * s))
                sy = int(round(cy + dy * s))
                if 0 <= sx < img_w and 0 <= sy < img_h:
                    total += float(img[sy, sx])
                    cnt += 1
            return total / cnt if cnt > 0 else 0.0

        supported = []
        sample_dist = 5

        for contour in (OUTER_LIP_INDICES, INNER_LIP_INDICES):
            n_pts = len(contour)
            for i, idx in enumerate(contour):
                px = int(round(pts[idx][0]))
                py = int(round(pts[idx][1]))
                if px < sample_dist or px >= w - sample_dist or py < sample_dist or py >= h - sample_dist:
                    supported.append(False)
                    continue

                prev_idx = contour[(i - 1) % n_pts]
                next_idx = contour[(i + 1) % n_pts]
                tx = pts[next_idx][0] - pts[prev_idx][0]
                ty = pts[next_idx][1] - pts[prev_idx][1]
                tlen = max((tx ** 2 + ty ** 2) ** 0.5, 1e-6)

                nx = -ty / tlen
                ny = tx / tlen

                outside = _sample_avg(clahe_gray, px, py, nx, ny, sample_dist, w, h)
                inside = _sample_avg(clahe_gray, px, py, -nx, -ny, sample_dist, w, h)
                contrast = abs(outside - inside)

                supported.append(contrast >= self._gradient_threshold)

        n_total = len(supported)
        if n_total == 0:
            return 0.0, 0

        visible_ratio = sum(supported) / n_total

        max_gap = 0
        current_gap = 0
        for s in supported + supported[:5]:
            if not s:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0

        return visible_ratio, max_gap

    # ------------------------------------------------------------------
    # Speaking expert
    # ------------------------------------------------------------------

    def _compute_speaking_prob(self, ts: _TrackMouthState) -> float:
        opens = list(ts.open_ratios)
        widths = list(ts.width_ratios)
        n = len(opens)
        if n < 3:
            return 0.0

        open_diffs = [opens[i + 1] - opens[i] for i in range(n - 1)]
        width_diffs = [widths[i + 1] - widths[i] for i in range(n - 1)]

        open_var = float(np.var(open_diffs)) if open_diffs else 0.0
        width_var = float(np.var(width_diffs)) if width_diffs else 0.0

        zero_crossings = sum(
            1 for i in range(len(open_diffs) - 1)
            if open_diffs[i] * open_diffs[i + 1] < 0
        )
        max_zc = max(len(open_diffs) - 1, 1)
        zc_ratio = zero_crossings / max_zc

        open_range = max(opens) - min(opens)

        s_open = min(open_var / 0.001, 1.0)
        s_width = min(width_var / 0.0005, 1.0)
        s_zc = min(zc_ratio / 0.4, 1.0)
        s_range = min(open_range / 0.15, 1.0)

        prob = 0.35 * s_open + 0.20 * s_width + 0.25 * s_zc + 0.20 * s_range
        return min(max(prob, 0.0), 1.0)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _get_track(self, track_id: int) -> _TrackMouthState:
        if track_id not in self._tracks:
            self._tracks[track_id] = _TrackMouthState(self._buffer_size)
        return self._tracks[track_id]

    def _get_pose_bucket(self, yaw: float) -> str:
        if abs(yaw) < 20:
            return "front"
        return "left" if yaw < 0 else "right"

    def _emit(
        self, ts: _TrackMouthState, status: str,
        speaking_prob: float, occlusion_prob: float,
        reason: str, timestamp_ms: float,
    ) -> MouthState:
        ts.status = status
        confidence = 0.8 if status not in ("unknown",) else 0.3
        return MouthState(
            status=status,
            speaking_prob=speaking_prob,
            occlusion_prob=occlusion_prob if "occlu" in status or "unobs" in status else 0.0,
            confidence=confidence,
            reason_code=reason,
            timestamp_ms=timestamp_ms,
        )

    def remove_track(self, track_id: int):
        self._tracks.pop(track_id, None)

    def reset(self):
        self._tracks.clear()

    def close(self):
        self._mesh.close()
