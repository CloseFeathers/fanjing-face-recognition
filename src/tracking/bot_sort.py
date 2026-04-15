"""
BoT-SORT Tracker (无 ReID / 无 CMC) —— 纯运动/几何关联。

核心算法: ByteTrack 风格双阶段匹配 + Kalman 滤波
  1) 高置信检测 vs 活跃轨迹 (IoU 匹配)
  2) 低置信检测 vs 剩余活跃轨迹 (IoU 匹配)
  3) 剩余高置信检测 vs 丢失轨迹 (重激活)
  4) 剩余高置信检测 → 新建轨迹

轨迹生命周期: tentative → confirmed → lost → removed
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..detectors.detection import FaceDetection
from ..ingestion.frame import Frame
from .kalman_filter import KalmanFilter
from .matching import associate
from .track import FrameTracks, STrack, TrackState

# ======================================================================
# 配置
# ======================================================================

@dataclass
class BoTSORTConfig:
    """Tracker 可调参数集 —— 四阈值分离架构。

    阈值 A~D (从低到高):
        detector_emit_min_det:  检测器输出到 tracker 的最低分数 (弱框放行)
        track_update_low_thres: 低分框可更新已有/丢失轨迹的阈值 (= A)
        track_new_high_thres:   允许创建新 track 的高分阈值 (只高分框可新建)
        sample_min_det:         进入 alignment/embedding 的采样阈值 (由 server 侧控制)

    其他:
        match_iou_threshold: IoU 匹配门槛
        min_hits:            tentative → confirmed 所需连续命中帧数
        max_age:             lost 状态最大存活检测帧数
    """
    detector_emit_min_det: float = 0.15
    track_new_high_thres: float = 0.50
    track_update_low_thres: float = 0.15
    match_iou_threshold: float = 0.3
    min_hits: int = 3
    max_age: int = 30


# ======================================================================
# Tracker
# ======================================================================

class BoTSORTTracker:
    """BoT-SORT 跟踪器 (无 ReID 版本, 四阈值分离)。"""

    def __init__(self, cfg: Optional[BoTSORTConfig] = None) -> None:
        self.cfg = cfg or BoTSORTConfig()
        self._kf = KalmanFilter()
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.frame_count: int = 0

        # 弱检测路径统计
        self.stats_raw_det_count: int = 0
        self.stats_above_emit: int = 0
        self.stats_above_high: int = 0
        self.stats_between_emit_and_high: int = 0
        self.stats_weak_used_update: int = 0

    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.frame_count = 0
        self.stats_raw_det_count = 0
        self.stats_above_emit = 0
        self.stats_above_high = 0
        self.stats_between_emit_and_high = 0
        self.stats_weak_used_update = 0
        STrack.reset_id()

    def get_det_stats(self) -> dict:
        return {
            "raw_det_count": self.stats_raw_det_count,
            "det_above_emit": self.stats_above_emit,
            "det_above_high": self.stats_above_high,
            "det_between_emit_and_high": self.stats_between_emit_and_high,
            "weak_used_update": self.stats_weak_used_update,
        }

    # ==================================================================
    # 主更新 (检测帧)
    # ==================================================================

    def update(
        self,
        detections: List[FaceDetection],
        img_w: int,
        img_h: int,
    ) -> List[STrack]:
        """用新检测更新 tracker，返回当前活跃轨迹。

        四阈值分离:
          A. detector_emit_min_det: 检测器已按此阈值输出
          B. track_new_high_thres:  只有高分框可新建轨迹
          C. track_update_low_thres: 低分框可更新已有/丢失轨迹
          D. sample_min_det: (server 侧控制, tracker 不涉及)
        """
        self.frame_count += 1
        cfg = self.cfg

        # ---- 统计: 原始检测数 ----
        self.stats_raw_det_count += len(detections)

        # ---- 0. 按 emit 阈值过滤 (正常情况 detector 已过滤, 这是安全兜底) ----
        dets = [d for d in detections if d.score >= cfg.detector_emit_min_det]
        self.stats_above_emit += len(dets)

        # ---- 0.5 清理 REMOVED 的轨迹 ----
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state != TrackState.REMOVED]
        self.lost_stracks = [t for t in self.lost_stracks if t.state != TrackState.REMOVED]

        # ---- 1. 预测所有轨迹 ----
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict(self._kf)
            t.age += 1

        # ---- 2. 按阈值 B 分高低 ----
        det_high = [d for d in dets if d.score >= cfg.track_new_high_thres]
        det_low = [d for d in dets if cfg.track_update_low_thres <= d.score < cfg.track_new_high_thres]
        self.stats_above_high += len(det_high)
        self.stats_between_emit_and_high += len(det_low)

        # ---- 3. 第一阶段: 高分检测 vs 活跃轨迹 ----
        trk_bboxes_1 = np.array(
            [t.bbox_xyxy_clipped(img_w, img_h) for t in self.tracked_stracks]
        ) if self.tracked_stracks else np.empty((0, 4))
        det_bboxes_1 = np.array(
            [d.bbox_xyxy for d in det_high]
        ) if det_high else np.empty((0, 4))

        matches_1, u_trk_1, u_det_1, ious_1 = associate(
            trk_bboxes_1, det_bboxes_1, cfg.match_iou_threshold
        )

        # ---- 4. 第二阶段: 低分检测 vs 未匹配活跃轨迹 ----
        remain_tracked = [self.tracked_stracks[i] for i in u_trk_1]
        trk_bboxes_2 = np.array(
            [t.bbox_xyxy_clipped(img_w, img_h) for t in remain_tracked]
        ) if remain_tracked else np.empty((0, 4))
        det_bboxes_2 = np.array(
            [d.bbox_xyxy for d in det_low]
        ) if det_low else np.empty((0, 4))

        matches_2, u_trk_2, u_det_low_unmatched, ious_2 = associate(
            trk_bboxes_2, det_bboxes_2, cfg.match_iou_threshold
        )

        # ---- 5. 第三阶段: 未匹配高分检测 vs 丢失轨迹 ----
        unmatched_high = [det_high[i] for i in u_det_1]
        lost_bboxes = np.array(
            [t.bbox_xyxy_clipped(img_w, img_h) for t in self.lost_stracks]
        ) if self.lost_stracks else np.empty((0, 4))
        uhd_bboxes = np.array(
            [d.bbox_xyxy for d in unmatched_high]
        ) if unmatched_high else np.empty((0, 4))

        matches_3, u_lost_after_high, u_det_3, ious_3 = associate(
            lost_bboxes, uhd_bboxes, cfg.match_iou_threshold
        )

        # ---- 6. 第四阶段: 未匹配低分检测 vs 仍未恢复的丢失轨迹 ----
        remaining_lost = [self.lost_stracks[i] for i in u_lost_after_high]
        unmatched_low = [det_low[i] for i in u_det_low_unmatched]

        rl_bboxes = np.array(
            [t.bbox_xyxy_clipped(img_w, img_h) for t in remaining_lost]
        ) if remaining_lost else np.empty((0, 4))
        ul_bboxes = np.array(
            [d.bbox_xyxy for d in unmatched_low]
        ) if unmatched_low else np.empty((0, 4))

        matches_4, u_lost_final, _, ious_4 = associate(
            rl_bboxes, ul_bboxes, cfg.match_iou_threshold
        )

        # ============================================================
        # 构建新的轨迹池
        # ============================================================
        new_tracked: List[STrack] = []
        new_lost: List[STrack] = []

        # -- 第一阶段匹配 (高分 → 活跃) --
        for (ti, di), iou in zip(matches_1, ious_1):
            trk = self.tracked_stracks[ti]
            d = det_high[di]
            trk.update(self._kf, d.bbox_xyxy, d.score, d.kps5, iou)
            if trk.state == TrackState.TENTATIVE and trk.hits >= cfg.min_hits:
                trk.state = TrackState.CONFIRMED
            new_tracked.append(trk)

        # -- 第二阶段匹配 (低分 → 活跃: 弱框维持轨迹) --
        for (rti, di), iou in zip(matches_2, ious_2):
            trk = remain_tracked[rti]
            d = det_low[di]
            trk.update(self._kf, d.bbox_xyxy, d.score, d.kps5, iou)
            self.stats_weak_used_update += 1
            new_tracked.append(trk)

        # -- 第三阶段匹配 (高分 → 丢失: 正式恢复) --
        for (li, di), iou in zip(matches_3, ious_3):
            trk = self.lost_stracks[li]
            d = unmatched_high[di]
            trk.re_activate(self._kf, d.bbox_xyxy, d.score, d.kps5, iou)
            new_tracked.append(trk)

        # -- 第四阶段匹配 (低分 → 丢失: 弱框续命丢失轨迹) --
        for (rli, di), iou in zip(matches_4, ious_4):
            trk = remaining_lost[rli]
            d = unmatched_low[di]
            trk.update(self._kf, d.bbox_xyxy, d.score, d.kps5, iou)
            self.stats_weak_used_update += 1
            new_lost.append(trk)

        # -- 未匹配的活跃轨迹 → lost / removed --
        truly_unmatched = [remain_tracked[i] for i in u_trk_2]
        for trk in truly_unmatched:
            trk.mark_missed()
            if trk.state == TrackState.TENTATIVE:
                trk.state = TrackState.REMOVED
            else:
                trk.state = TrackState.LOST
                new_lost.append(trk)

        # -- 未匹配的丢失轨迹 → 继续 lost / removed --
        for i in u_lost_final:
            trk = remaining_lost[i]
            trk.mark_missed()
            if trk.time_since_update > cfg.max_age:
                trk.state = TrackState.REMOVED
            else:
                new_lost.append(trk)

        # -- 新建轨迹: 只有高分框可以创建 --
        remaining_high = [unmatched_high[i] for i in u_det_3]
        for d in remaining_high:
            if d.score >= cfg.track_new_high_thres:
                t = STrack.from_detection(d.bbox_xyxy, d.score, d.kps5)
                t.activate(self._kf)
                new_tracked.append(t)

        # ---- 更新轨迹池 ----
        self.tracked_stracks = new_tracked
        self.lost_stracks = new_lost

        return [t for t in new_tracked + new_lost
                if t.state != TrackState.REMOVED]

    # ==================================================================
    # 仅预测 (非检测帧, 用于 det_every_n > 1)
    # ==================================================================

    def predict_only(self, img_w: int, img_h: int) -> List[STrack]:
        """仅 Kalman 预测，不做匹配/状态转换。"""
        self.frame_count += 1
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict(self._kf)
            t.age += 1
            # 注意: 不增加 time_since_update (无检测机会)
            # 清空本帧的检测关联信息
            t.det_score = None
            t.kps5 = None
            t.match_iou = None

        return [t for t in self.tracked_stracks + self.lost_stracks
                if t.state != TrackState.REMOVED]

    # ==================================================================
    # 完整管线入口
    # ==================================================================

    def step(
        self,
        frame: Frame,
        detector,
        do_detect: bool = True,
    ) -> FrameTracks:
        """一帧完整处理: 检测(可选) + 跟踪。

        Args:
            frame:     Ingestion 输出帧
            detector:  SCRFDDetector 实例
            do_detect: 是否执行检测 (由 det_every_n 控制)
        Returns:
            FrameTracks
        """
        W, H = frame.width, frame.height

        detect_ms = 0.0
        if do_detect:
            dets_result = detector.detect(frame)
            detect_ms = dets_result.detect_time_ms
            det_faces = dets_result.faces
        else:
            det_faces = []

        t0 = time.perf_counter()
        if do_detect:
            active = self.update(det_faces, W, H)
        else:
            active = self.predict_only(W, H)
        track_ms = (time.perf_counter() - t0) * 1000.0

        # 保存原始 STrack 列表供外部模块 (对齐/采样) 使用
        self._last_active = list(active)

        tracks_out = [t.to_dict(W, H) for t in active]

        return FrameTracks(
            timestamp_ms=frame.timestamp_ms,
            frame_id=frame.frame_id,
            source_id=frame.source_id,
            width=W,
            height=H,
            num_tracks=len(tracks_out),
            did_detect=do_detect,
            detect_time_ms=detect_ms,
            track_time_ms=track_ms,
            tracks=tracks_out,
        )

    @property
    def last_active_stracks(self) -> List[STrack]:
        """最近一次 step() 产生的原始 STrack 列表。"""
        return getattr(self, "_last_active", [])
