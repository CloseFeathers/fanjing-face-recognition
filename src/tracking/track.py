"""
Track data structures — STrack (internal track) and FrameTracks (frame-level output contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .kalman_filter import KalmanFilter

# ======================================================================
# Track State
# ======================================================================

class TrackState:
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


# ======================================================================
# bbox format conversion
# ======================================================================

def xyxy_to_cxywh(bbox) -> np.ndarray:
    b = np.asarray(bbox, dtype=np.float64)
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    w = b[2] - b[0]
    h = b[3] - b[1]
    return np.array([cx, cy, w, h], dtype=np.float64)


def cxywh_to_xyxy(cxywh) -> np.ndarray:
    cx, cy, w, h = cxywh[0], cxywh[1], cxywh[2], cxywh[3]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    dtype=np.float64)


# ======================================================================
# STrack — Single track (tracker internal use)
# ======================================================================

class STrack:
    """Single tracking trajectory, holds its own Kalman state."""

    _next_id: int = 0

    @classmethod
    def reset_id(cls) -> None:
        cls._next_id = 0

    @classmethod
    def _alloc_id(cls) -> int:
        cls._next_id += 1
        return cls._next_id

    def __init__(self) -> None:
        self.track_id: int = 0
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.det_score: Optional[float] = None
        self.kps5: Optional[List[List[float]]] = None
        self.state: str = TrackState.TENTATIVE

        self.age: int = 0
        self.hits: int = 0
        self.time_since_update: int = 0
        self.match_iou: Optional[float] = None

        # ======== Credit Gate (credit score system) ========
        self.face_valid_credit: float = 0.0
        self.ever_sampled: bool = False
        self.linked_person: bool = False

    # --- Lifecycle ---

    @classmethod
    def from_detection(cls, bbox_xyxy, score, kps5=None) -> "STrack":
        t = cls()
        t._bbox_init = np.asarray(bbox_xyxy, dtype=np.float64)
        t.det_score = float(score)
        t.kps5 = kps5
        return t

    def activate(self, kf: KalmanFilter) -> None:
        """First activation, allocate track_id and initialize Kalman state."""
        self.track_id = self._alloc_id()
        measurement = xyxy_to_cxywh(self._bbox_init)
        self.mean, self.covariance = kf.initiate(measurement)
        self.state = TrackState.TENTATIVE
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def predict(self, kf: KalmanFilter) -> None:
        """Kalman predict (does not update time_since_update)."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

    def update(self, kf: KalmanFilter, bbox_xyxy, score,
               kps5=None, iou: Optional[float] = None) -> None:
        """Kalman update after matching with detection."""
        measurement = xyxy_to_cxywh(bbox_xyxy)
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, measurement
        )
        self.det_score = float(score)
        self.kps5 = kps5
        self.hits += 1
        self.time_since_update = 0
        self.match_iou = iou

    def re_activate(self, kf: KalmanFilter, bbox_xyxy, score,
                    kps5=None, iou: Optional[float] = None) -> None:
        """Reactivate lost track (keep original track_id)."""
        measurement = xyxy_to_cxywh(bbox_xyxy)
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, measurement
        )
        self.det_score = float(score)
        self.kps5 = kps5
        self.state = TrackState.CONFIRMED
        self.hits += 1
        self.time_since_update = 0
        self.match_iou = iou

    def mark_missed(self) -> None:
        """No detection matched this frame (only called on detection frames)."""
        self.time_since_update += 1
        self.det_score = None
        self.kps5 = None
        self.match_iou = None

    # --- bbox properties ---

    @property
    def bbox_xyxy(self) -> np.ndarray:
        return cxywh_to_xyxy(self.mean[:4])

    def bbox_xyxy_clipped(self, img_w: int, img_h: int) -> List[float]:
        b = self.bbox_xyxy
        return [
            float(max(0.0, min(b[0], img_w))),
            float(max(0.0, min(b[1], img_h))),
            float(max(0.0, min(b[2], img_w))),
            float(max(0.0, min(b[3], img_h))),
        ]

    # --- Serialization ---

    def to_dict(self, img_w: int, img_h: int) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "track_id": self.track_id,
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy_clipped(img_w, img_h)],
            "det_score": round(self.det_score, 4) if self.det_score is not None else None,
            "state": self.state,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "match_iou": round(self.match_iou, 4) if self.match_iou is not None else None,
            "face_valid_credit": round(self.face_valid_credit, 2),
            "ever_sampled": self.ever_sampled,
            "linked_person": self.linked_person,
        }
        return d


# ======================================================================
# FrameTracks — Frame-level output
# ======================================================================

@dataclass(slots=True)
class FrameTracks:
    timestamp_ms: float
    frame_id: int
    source_id: str
    width: int
    height: int
    num_tracks: int = 0
    did_detect: bool = True
    detect_time_ms: float = 0.0
    track_time_ms: float = 0.0
    tracks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": round(self.timestamp_ms, 3),
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "width": self.width,
            "height": self.height,
            "num_tracks": self.num_tracks,
            "did_detect": self.did_detect,
            "detect_time_ms": round(self.detect_time_ms, 2),
            "track_time_ms": round(self.track_time_ms, 2),
            "tracks": self.tracks,
        }
