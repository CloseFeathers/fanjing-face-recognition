"""
MediaPipe FaceLandmarker 封装 — 从 face crop 提取 478 关键点及嘴部几何特征。

使用 MediaPipe Tasks API (v0.10.31+), 替代已废弃的 solutions API。

输入: BGR face crop (从原始帧按 bbox 裁切的高分辨率人脸图)
输出: MeshResult (478 landmarks + 嘴部几何特征 + 头部姿态估计)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ======================================================================
# 嘴部关键点索引 (MediaPipe 478-point Face Mesh)
# ======================================================================

OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                     409, 270, 269, 267, 0, 37, 39, 40, 185]

INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                     415, 310, 311, 312, 13, 82, 81, 80, 191]

UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
UPPER_LIP_OUTER = 0
LOWER_LIP_OUTER = 17

LEFT_FACE_CONTOUR = [234, 93, 132]
RIGHT_FACE_CONTOUR = [454, 323, 361]
NOSE_TIP = 1
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362

ALL_MOUTH_INDICES = sorted(set(OUTER_LIP_INDICES + INNER_LIP_INDICES))


@dataclass
class MeshResult:
    """单帧 Face Mesh 分析结果。"""
    landmarks_478: Optional[np.ndarray]

    mouth_open_ratio: float
    mouth_width_ratio: float
    lip_thickness: float

    head_yaw: float

    mouth_landmarks: Optional[np.ndarray]

    face_crop_size: float
    mesh_confidence: float

    mouth_shape_score: float


class MeshDetector:
    """MediaPipe FaceLandmarker 推理封装 (Tasks API)。"""

    def __init__(self, model_path: str = "models/face_landmarker.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FaceLandmarker model not found: {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, face_crop_bgr: np.ndarray) -> MeshResult:
        """对 face crop 做 FaceLandmarker 推理。"""
        h, w = face_crop_bgr.shape[:2]
        face_crop_size = float(min(h, w))

        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return self._empty_result(face_crop_size)

        face_lm = result.face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in face_lm],
                       dtype=np.float32)

        mouth_pts = pts[ALL_MOUTH_INDICES]

        mouth_open_ratio = self._compute_mouth_open_ratio(pts)
        mouth_width_ratio = self._compute_mouth_width_ratio(pts)
        lip_thickness = self._compute_lip_thickness(pts)

        head_yaw = 0.0
        if result.facial_transformation_matrixes:
            head_yaw = self._yaw_from_matrix(result.facial_transformation_matrixes[0])
        else:
            head_yaw = self._estimate_yaw_fallback(pts)

        mouth_shape_score = self._compute_mouth_shape_score(pts)

        return MeshResult(
            landmarks_478=pts,
            mouth_open_ratio=mouth_open_ratio,
            mouth_width_ratio=mouth_width_ratio,
            lip_thickness=lip_thickness,
            head_yaw=head_yaw,
            mouth_landmarks=mouth_pts,
            face_crop_size=face_crop_size,
            mesh_confidence=1.0,
            mouth_shape_score=mouth_shape_score,
        )

    def close(self) -> None:
        """释放 MediaPipe 资源。"""
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __del__(self) -> None:
        """析构时确保资源释放。"""
        self.close()

    def __enter__(self) -> "MeshDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------------

    def _compute_mouth_open_ratio(self, pts: np.ndarray) -> float:
        upper = pts[UPPER_LIP_TOP][:2]
        lower = pts[LOWER_LIP_BOTTOM][:2]
        left = pts[LEFT_MOUTH_CORNER][:2]
        right = pts[RIGHT_MOUTH_CORNER][:2]
        open_dist = float(np.linalg.norm(lower - upper))
        width = float(np.linalg.norm(right - left))
        return open_dist / width if width > 1.0 else 0.0

    def _compute_mouth_width_ratio(self, pts: np.ndarray) -> float:
        left_mouth = pts[LEFT_MOUTH_CORNER][:2]
        right_mouth = pts[RIGHT_MOUTH_CORNER][:2]
        left_eye = pts[LEFT_EYE_INNER][:2]
        right_eye = pts[RIGHT_EYE_INNER][:2]
        mouth_w = float(np.linalg.norm(right_mouth - left_mouth))
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        return mouth_w / eye_dist if eye_dist > 1.0 else 0.0

    def _compute_lip_thickness(self, pts: np.ndarray) -> float:
        upper_outer = pts[UPPER_LIP_OUTER][:2]
        upper_inner = pts[UPPER_LIP_TOP][:2]
        lower_inner = pts[LOWER_LIP_BOTTOM][:2]
        lower_outer = pts[LOWER_LIP_OUTER][:2]
        upper_thick = float(np.linalg.norm(upper_inner - upper_outer))
        lower_thick = float(np.linalg.norm(lower_outer - lower_inner))
        left = pts[LEFT_MOUTH_CORNER][:2]
        right = pts[RIGHT_MOUTH_CORNER][:2]
        width = float(np.linalg.norm(right - left))
        return (upper_thick + lower_thick) / width if width > 1.0 else 0.0

    @staticmethod
    def _yaw_from_matrix(matrix) -> float:
        """从 MediaPipe 输出的 4x4 面部变换矩阵提取 yaw 角度。"""
        try:
            m = np.array(matrix).reshape(4, 4) if not isinstance(matrix, np.ndarray) else matrix
            r00, r02 = float(m[0, 0]), float(m[0, 2])
            yaw = float(np.degrees(np.arctan2(r02, r00)))
            return yaw
        except (ValueError, TypeError, IndexError):
            return 0.0

    def _estimate_yaw_fallback(self, pts: np.ndarray) -> float:
        """后备: 用鼻尖-脸宽比估算 yaw（变换矩阵不可用时）。"""
        nose_x = pts[NOSE_TIP][0]
        left_x = np.mean([pts[i][0] for i in LEFT_FACE_CONTOUR])
        right_x = np.mean([pts[i][0] for i in RIGHT_FACE_CONTOUR])
        face_w = right_x - left_x
        if face_w < 1.0:
            return 0.0
        nose_ratio = (nose_x - left_x) / face_w
        asymmetry = (nose_ratio - 0.5) * 2.0
        return float(asymmetry * 90.0)

    def _compute_mouth_shape_score(self, pts: np.ndarray) -> float:
        mouth_2d = pts[ALL_MOUTH_INDICES][:, :2]
        xs, ys = mouth_2d[:, 0], mouth_2d[:, 1]
        bbox_w = xs.max() - xs.min()
        bbox_h = ys.max() - ys.min()
        bbox_area = bbox_w * bbox_h
        if bbox_area < 1.0:
            return 0.0
        try:
            hull = cv2.convexHull(mouth_2d.astype(np.float32))
            hull_area = float(cv2.contourArea(hull))
        except cv2.error:
            return 0.0
        area_ratio = hull_area / bbox_area
        area_score = min(area_ratio / 0.6, 1.0)

        center_x = (xs.max() + xs.min()) / 2
        left_pts = mouth_2d[mouth_2d[:, 0] < center_x]
        right_pts = mouth_2d[mouth_2d[:, 0] >= center_x]
        sym_score = 1.0
        if len(left_pts) > 0 and len(right_pts) > 0:
            left_spread = np.std(left_pts[:, 1])
            right_spread = np.std(right_pts[:, 1])
            if max(left_spread, right_spread) > 0:
                sym_score = min(left_spread, right_spread) / max(left_spread, right_spread)

        return 0.6 * area_score + 0.4 * sym_score

    def _empty_result(self, face_crop_size: float) -> MeshResult:
        return MeshResult(
            landmarks_478=None,
            mouth_open_ratio=0.0,
            mouth_width_ratio=0.0,
            lip_thickness=0.0,
            head_yaw=0.0,
            mouth_landmarks=None,
            face_crop_size=face_crop_size,
            mesh_confidence=0.0,
            mouth_shape_score=0.0,
        )
