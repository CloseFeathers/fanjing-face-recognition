"""
Detection data structures — Unified output contract for face detection.

FrameDetections is the complete detection result for a frame, directly serializable as one JSONL line.
FaceDetection is the detection info for a single face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class FaceDetection:
    """Single face detection result.

    Attributes:
        bbox_xyxy: Pixel coordinates [x1, y1, x2, y2], guaranteed x1<x2, y1<y2
        score:     Detection confidence (0~1)
        class_id:  Fixed to 0 for face
        kps5:      5 keypoints [[x,y], ...] in order:
                   left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner
                   None if model doesn't output keypoints
    """

    bbox_xyxy: List[float]
    score: float
    class_id: int = 0
    kps5: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy],
            "score": round(self.score, 4),
            "class_id": self.class_id,
        }
        if self.kps5 is not None:
            d["kps5"] = [[round(x, 2), round(y, 2)] for x, y in self.kps5]
        return d


@dataclass(slots=True)
class FrameDetections:
    """All face detection results for one frame.

    Attributes:
        timestamp_ms:   Timestamp from Ingestion Frame
        frame_id:       Frame sequence number, aligned with Ingestion
        source_id:      Data source identifier
        width:          Frame width (pixels)
        height:         Frame height (pixels)
        num_faces:      Number of detected faces
        detect_time_ms: Detection time for this frame (milliseconds)
        faces:          Face list
    """

    timestamp_ms: float
    frame_id: int
    source_id: str
    width: int
    height: int
    num_faces: int = 0
    detect_time_ms: float = 0.0
    faces: List[FaceDetection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": round(self.timestamp_ms, 3),
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "width": self.width,
            "height": self.height,
            "num_faces": self.num_faces,
            "detect_time_ms": round(self.detect_time_ms, 2),
            "faces": [f.to_dict() for f in self.faces],
        }
