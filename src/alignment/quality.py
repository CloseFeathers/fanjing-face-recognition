"""
Quality Gate —— 人脸质量评估。

评估维度:
  1. det_score     检测置信度 (来自 SCRFD 真实输出)
  2. bbox_area     人脸 bbox 像素面积 (来自真实 bbox_xyxy)
  3. blur_score    清晰度 (在对齐后的 112x112 图上计算 Laplacian 方差)
  4. kps_validity  关键点几何合理性

流程: 先调用 aligner.align() 得到对齐图, 再调用 evaluate()。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class QualityResult:
    """单次质量评估结果。"""
    passed: bool
    score: float               # 综合质量分 [0, 1]
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
    """质量门槛配置。

    参数说明:
        min_det_score:  最低检测分数
        min_bbox_area:  最小 bbox 像素面积 (w*h)
        min_blur_score: 最低清晰度 (对齐后 112x112 图的 Laplacian 方差)
        min_eye_dist:   最短双眼距离像素
    """
    min_det_score: float = 0.60
    min_bbox_area: float = 900.0
    min_blur_score: float = 40.0
    min_eye_dist: float = 20.0


class QualityGate:
    """人脸质量门控器。"""

    def __init__(self, cfg: Optional[QualityConfig] = None) -> None:
        self.cfg = cfg or QualityConfig()

    def evaluate(
        self,
        aligned_face: np.ndarray,
        bbox_xyxy: List[float],
        det_score: float,
        kps5: Optional[List[List[float]]],
    ) -> QualityResult:
        """评估一张人脸的质量。

        Args:
            aligned_face: 对齐后的 112x112 BGR 图 (来自 warpAffine 真实输出)
            bbox_xyxy:    原始人脸框 [x1, y1, x2, y2]
            det_score:    SCRFD 检测置信度 (真实值)
            kps5:         5 点关键点
        """
        cfg = self.cfg
        reasons: List[str] = []

        # ---- 1. det_score (来自 SCRFD 真实输出) ----
        if det_score < cfg.min_det_score:
            reasons.append(f"det_score={det_score:.2f}<{cfg.min_det_score}")

        # ---- 2. bbox_area (来自真实 bbox) ----
        x1, y1, x2, y2 = bbox_xyxy
        w = max(x2 - x1, 0)
        h = max(y2 - y1, 0)
        area = w * h
        if area < cfg.min_bbox_area:
            reasons.append(f"area={area:.0f}<{cfg.min_bbox_area}")

        # ---- 3. blur (在对齐后 112x112 图上计算 Laplacian 方差) ----
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
    """计算图像的 Laplacian 方差 (越大越清晰)。"""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _check_kps(kps5, bbox_xyxy, cfg) -> bool:
    """检查 5 点关键点的几何合理性。"""
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
