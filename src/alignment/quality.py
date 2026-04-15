"""
Quality Gate — Face quality assessment.

Assessment dimensions:
  1. det_score     Detection confidence (from SCRFD actual output)
  2. bbox_area     Face bbox pixel area (from actual bbox_xyxy)
  3. blur_score    Sharpness (Laplacian variance computed on aligned 112x112 image)
  4. kps_validity  Keypoint geometric validity

Flow: First call aligner.align() to get aligned image, then call evaluate().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class QualityResult:
    """Single quality assessment result."""
    passed: bool
    score: float               # Composite quality score [0, 1]
    det_score: float
    bbox_area: float
    blur_score: float
    kps_valid: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_passed": self.passed,
            "quality_score": round(self.score, 4),
            "det_score": round(self.det_score, 4),
            "bbox_area": round(self.bbox_area, 1),
            "blur_score": round(self.blur_score, 1),
            "kps_valid": self.kps_valid,
            "reject_reasons": self.reasons if not self.passed else [],
        }


@dataclass
class QualityConfig:
    """Quality threshold configuration.

    Parameters:
        min_det_score:  Minimum detection score
        min_bbox_area:  Minimum bbox pixel area (w*h)
        min_blur_score: Minimum sharpness (Laplacian variance of aligned 112x112 image)
        min_eye_dist:   Minimum eye distance in pixels
    """
    min_det_score: float = 0.60
    min_bbox_area: float = 900.0
    min_blur_score: float = 40.0
    min_eye_dist: float = 20.0


class QualityGate:
    """Face quality gate controller."""

    def __init__(self, cfg: Optional[QualityConfig] = None) -> None:
        self.cfg = cfg or QualityConfig()

    def evaluate(
        self,
        aligned_face: np.ndarray,
        bbox_xyxy: List[float],
        det_score: float,
        kps5: Optional[List[List[float]]],
    ) -> QualityResult:
        """Evaluate face quality.

        Args:
            aligned_face: Aligned 112x112 BGR image (from warpAffine actual output)
            bbox_xyxy:    Original face box [x1, y1, x2, y2]
            det_score:    SCRFD detection confidence (actual value)
            kps5:         5-point keypoints
        """
        cfg = self.cfg
        reasons: List[str] = []

        # ---- 1. det_score (from SCRFD actual output) ----
        if det_score < cfg.min_det_score:
            reasons.append(f"det_score={det_score:.2f}<{cfg.min_det_score}")

        # ---- 2. bbox_area (from actual bbox) ----
        x1, y1, x2, y2 = bbox_xyxy
        w = max(x2 - x1, 0)
        h = max(y2 - y1, 0)
        area = w * h
        if area < cfg.min_bbox_area:
            reasons.append(f"area={area:.0f}<{cfg.min_bbox_area}")

        # ---- 3. blur (compute Laplacian variance on aligned 112x112 image) ----
        blur_score = compute_blur(aligned_face)
        if blur_score < cfg.min_blur_score:
            reasons.append(f"blur={blur_score:.1f}<{cfg.min_blur_score}")

        # ---- 4. kps validity ----
        kps_valid = _check_kps(kps5, bbox_xyxy, cfg)
        if not kps_valid:
            reasons.append("kps_invalid")

        passed = len(reasons) == 0

        score = (
            0.30 * min(det_score / 1.0, 1.0)
            + 0.20 * min(area / 15000.0, 1.0)
            + 0.30 * min(blur_score / 200.0, 1.0)
            + 0.20 * (1.0 if kps_valid else 0.0)
        )

        return QualityResult(
            passed=passed,
            score=score,
            det_score=det_score,
            bbox_area=area,
            blur_score=blur_score,
            kps_valid=kps_valid,
            reasons=reasons,
        )


def compute_blur(image: np.ndarray) -> float:
    """Compute image Laplacian variance (higher = sharper)."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _check_kps(kps5, bbox_xyxy, cfg) -> bool:
    """Check 5-point keypoint geometric validity."""
    if kps5 is None or len(kps5) < 5:
        return False

    pts = np.array(kps5, dtype=np.float32)
    if pts.shape != (5, 2):
        return False

    x1, y1, x2, y2 = bbox_xyxy

    margin = max(x2 - x1, y2 - y1) * 0.15
    for px, py in pts:
        if px < x1 - margin or px > x2 + margin:
            return False
        if py < y1 - margin or py > y2 + margin:
            return False

    left_eye, right_eye = pts[0], pts[1]
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < cfg.min_eye_dist:
        return False

    return True
