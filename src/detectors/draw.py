"""
Detection visualization — Draw bbox, confidence, keypoints, and detection HUD on frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detection import FrameDetections

# Keypoint colors (BGR): left_eye, right_eye, nose, left_mouth, right_mouth
KPS_COLORS = [
    (255, 0, 0),     # Blue - left eye
    (0, 0, 255),     # Red - right eye
    (0, 255, 0),     # Green - nose tip
    (255, 255, 0),   # Cyan - left mouth corner
    (0, 255, 255),   # Yellow - right mouth corner
]


def draw_detections(
    image: np.ndarray,
    dets: FrameDetections,
    pipeline_fps: float,
    dropped_frames: int = 0,
) -> np.ndarray:
    """Overlay detection boxes, keypoints and HUD on frame, return new image."""
    img = image.copy()

    # ------------------------------------------------------------------
    # 1. Draw each face
    # ------------------------------------------------------------------
    for face in dets.faces:
        x1, y1, x2, y2 = [int(round(v)) for v in face.bbox_xyxy]

        # bbox rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Confidence label
        label = f"{face.score:.2f}"
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
        )
        # Label background
        cv2.rectangle(
            img,
            (x1, y1 - label_size[1] - baseline - 4),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            img, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA,
        )

        # 5-point keypoints
        if face.kps5 is not None:
            for idx, (px, py) in enumerate(face.kps5):
                color = KPS_COLORS[idx] if idx < len(KPS_COLORS) else (255, 255, 255)
                cv2.circle(img, (int(round(px)), int(round(py))), 3, color, -1)

    # ------------------------------------------------------------------
    # 2. HUD info panel (top-left)
    # ------------------------------------------------------------------
    lines = [
        f"source  : {dets.source_id}",
        f"frame   : {dets.frame_id}",
        f"ts(ms)  : {dets.timestamp_ms:.1f}",
        f"size    : {dets.width}x{dets.height}",
        f"FPS     : {pipeline_fps:.1f}",
        f"det(ms) : {dets.detect_time_ms:.1f}",
        f"faces   : {dets.num_faces}",
        f"dropped : {dropped_frames}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    line_height = 22
    padding = 8

    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, tw)

    panel_w = max_w + padding * 2
    panel_h = line_height * len(lines) + padding * 2

    # Semi-transparent black background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    y = padding + 15
    for line in lines:
        cv2.putText(img, line, (padding, y), font, font_scale,
                    (0, 255, 0), thickness, cv2.LINE_AA)
        y += line_height

    return img
