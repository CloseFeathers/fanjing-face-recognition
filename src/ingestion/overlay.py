"""
Preview overlay — Render frame metadata HUD on the image.

Overlay content: timestamp_ms / frame_id / source_id / FPS / dropped_frames
"""

from __future__ import annotations

import cv2
import numpy as np

from .frame import Frame


def draw_overlay(
    image: np.ndarray,
    frame: Frame,
    fps: float,
) -> np.ndarray:
    """Overlay semi-transparent info panel on top-left, returns new image (doesn't modify original)."""
    img = image.copy()

    lines = [
        f"source : {frame.source_id}",
        f"frame  : {frame.frame_id}",
        f"ts(ms) : {frame.timestamp_ms:.1f}",
        f"size   : {frame.width}x{frame.height}",
        f"FPS    : {fps:.1f}",
        f"dropped: {frame.dropped_frames}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    color = (0, 255, 0)       # Green
    bg_color = (0, 0, 0)      # Black background
    line_height = 22
    padding = 8

    # Calculate panel dimensions
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, tw)

    panel_w = max_w + padding * 2
    panel_h = line_height * len(lines) + padding * 2

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Text
    y = padding + 16
    for line in lines:
        cv2.putText(img, line, (padding, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height

    return img
