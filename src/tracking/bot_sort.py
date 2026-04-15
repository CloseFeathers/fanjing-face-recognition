"""
BoT-SORT Tracker (without ReID / without CMC) — Pure motion/geometric association.

Core algorithm: ByteTrack-style two-stage matching + Kalman filtering
  1) High-confidence detections vs active tracks (IoU matching)
  2) Low-confidence detections vs remaining active tracks (IoU matching)
  3) Remaining high-confidence detections vs lost tracks (reactivation)
  4) Remaining high-confidence detections → create new tracks

Track lifecycle: tentative → confirmed → lost → removed
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
# Configuration
# ======================================================================

@dataclass
class BoTSORTConfig:
    """Tracker configurable parameter set — Four-threshold separation architecture.

    Thresholds A~D (low to high):
        detector_emit_min_det:  Minimum score for detector output to tracker (weak box pass-through)
        track_update_low_thres: Threshold for low-score boxes to update existing/lost tracks (= A)
        track_new_high_thres:   High-score threshold for creating new tracks (only high-score can create)
        sample_min_det:         Sampling threshold for alignment/embedding (controlled by server side)

    Others:
        match_iou_threshold: IoU matching threshold
        min_hits:            Consecutive hit frames required for tentative → confirmed
        max_age:             Maximum detection frames for lost state survival
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
    """BoT-SORT tracker (without ReID version, four-threshold separation)."""

    def __init__(self, cfg: Optional[BoTSORTConfig] = None) -> None:
        self.cfg = cfg or BoTSORTConfig()
        self._kf = KalmanFilter()
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.frame_count: int = 0

        # Weak detection path statistics
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
    # Main update (detection frame)
    # ==================================================================

    def update(
        self,
        detections: List[FaceDetection],
        img_w: int,
        img_h: int,
    ) -> List[STrack]:
        """Update tracker with new detections, return current active tracks.

        Four-threshold separation:
          A. detector_emit_min_det: Detector already filtered by this threshold
          B. track_new_high_thres:  Only high-score boxes can create new tracks
          C. track_update_low_thres: Low-score boxes can update existing/lost tracks
          D. sample_min_det: (Server-side controlled, tracker not involved)
        """
        self.frame_count += 1
        cfg = self.cfg

        # ---- Statistics: raw detection count ----
        self.stats_raw_det_count += len(detections)

        # ---- 0. Filter by emit threshold (normally detector already filtered, this is safety fallback) ----
        dets = [d for d in detections if d.score >= cfg.detector_emit_min_det]
        self.stats_above_emit += len(dets)

        # ---- 0.5 Clean up REMOVED tracks ----
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state != TrackState.REMOVED]
        self.lost_stracks = [t for t in self.lost_stracks if t.state != TrackState.REMOVED]

        # ---- 1. Predict all tracks ----
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict(self._kf)
            t.age += 1

        # ---- 2. Split by threshold B into high/low ----
        det_high = [d for d in dets if d.score >= cfg.track_new_high_thres]
        det_low = [d for d in dets if cfg.track_update_low_thres <= d.score < cfg.track_new_high_thres]
        self.stats_above_high += len(det_high)
        self.stats_between_emit_and_high += len(det_low)

        # ---- 3. Stage 1: High-score detections vs active tracks ----
        trk_bboxes_1 = np.array(
            [t.bbox_xyxy_clipped(img_w, img_h) for t in self.tracked_stracks]
        ) if self.tracked_stracks else np.empty((0, 4))
        det_bboxes_1 = np.array(
            [d.bbox_xyxy for d in det_high]
        ) if det_high else np.empty((0, 4))

        matches_1, u_trk_1, u_det_1, ious_1 = associate(
            trk_bboxes_1, det_bboxes_1, cfg.match_iou_threshold
        )

        # ---- 4. Stage 2: Low-score detections vs unmatched active tracks ----
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

        # ---- 5. Stage 3: Unmatched high-score detections vs lost tracks ----
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

        # ---- 6. Stage 4: Unmatched low-score detections vs still unrecovered lost tracks ----
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
        # Build new track pool
        # ============================================================
        new_tracked: List[STrack] = []
        new_lost: List[STrack] = []

        # -- Stage 1 matches (high → active) --
        for (ti, di), iou in zip(matches_1, ious_1):
            trk = self.tracked_stracks[ti]
            d = det_high[di]
            trk.update(self._kf, d.bbox_xyxy, d.score, d.kps5, iou)
            if trk.state == TrackState.TENTATIVE and trk.hits >= cfg.min_hits:
                trk.state = TrackState.CONFIRMED
            new_tracked.append(trk)

        # -- Stage 2 matches (low → active: weak box maintains track) --
        for (rti, di), iou in zip(matches_2, ious_2):
            trk = remain_tracked[rti]
            d = det_low[di]
            trk.update(self._kf, d.bbox_xyxy, d.score, d.kps5, iou)
            self.stats_weak_used_update += 1
            new_tracked.append(trk)

        # -- Stage 3 matches (high → lost: official recovery) --
        for (li, di), iou in zip(matches_3, ious_3):
            trk = self.lost_stracks[li]
            d = unmatched_high[di]
            trk.re_activate(self._kf, d.bbox_xyxy, d.score, d.kps5, iou)
            new_tracked.append(trk)

        # -- Stage 4 matches (low → lost: weak box extends lost track) --
        for (rli, di), iou in zip(matches_4, ious_4):
            trk = remaining_lost[rli]
            d = unmatched_low[di]
            trk.update(self._kf, d.bbox_xyxy, d.score, d.kps5, iou)
            self.stats_weak_used_update += 1
            new_lost.append(trk)

        # -- Unmatched active tracks → lost / removed --
        truly_unmatched = [remain_tracked[i] for i in u_trk_2]
        for trk in truly_unmatched:
            trk.mark_missed()
            if trk.state == TrackState.TENTATIVE:
                trk.state = TrackState.REMOVED
            else:
                trk.state = TrackState.LOST
                new_lost.append(trk)

        # -- Unmatched lost tracks → continue lost / removed --
        for i in u_lost_final:
            trk = remaining_lost[i]
            trk.mark_missed()
            if trk.time_since_update > cfg.max_age:
                trk.state = TrackState.REMOVED
            else:
                new_lost.append(trk)

        # -- Create new tracks: only high-score boxes can create --
        remaining_high = [unmatched_high[i] for i in u_det_3]
        for d in remaining_high:
            if d.score >= cfg.track_new_high_thres:
                t = STrack.from_detection(d.bbox_xyxy, d.score, d.kps5)
                t.activate(self._kf)
                new_tracked.append(t)

        # ---- Update track pool ----
        self.tracked_stracks = new_tracked
        self.lost_stracks = new_lost

        return [t for t in new_tracked + new_lost
                if t.state != TrackState.REMOVED]

    # ==================================================================
    # Predict only (non-detection frame, for det_every_n > 1)
    # ==================================================================

    def predict_only(self, img_w: int, img_h: int) -> List[STrack]:
        """Kalman predict only, no matching/state transitions."""
        self.frame_count += 1
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict(self._kf)
            t.age += 1
            # Note: don't increment time_since_update (no detection opportunity)
            # Clear detection association info for this frame
            t.det_score = None
            t.kps5 = None
            t.match_iou = None

        return [t for t in self.tracked_stracks + self.lost_stracks
                if t.state != TrackState.REMOVED]

    # ==================================================================
    # Full pipeline entry
    # ==================================================================

    def step(
        self,
        frame: Frame,
        detector,
        do_detect: bool = True,
    ) -> FrameTracks:
        """Full frame processing: detection (optional) + tracking.

        Args:
            frame:     Ingestion output frame
            detector:  SCRFDDetector instance
            do_detect: Whether to execute detection (controlled by det_every_n)
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

        # Save raw STrack list for external modules (alignment/sampling)
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
        """Raw STrack list from most recent step()."""
        return getattr(self, "_last_active", [])
