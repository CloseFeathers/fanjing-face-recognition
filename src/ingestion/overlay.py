"""
预览叠加层 —— 在画面上渲染帧元数据 HUD。

叠加内容：timestamp_ms / frame_id / source_id / FPS / dropped_frames
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
    """在图像左上角叠加半透明信息面板，返回新图像（不修改原图）。"""
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
    color = (0, 255, 0)       # 绿色
    bg_color = (0, 0, 0)      # 黑色背景
    line_height = 22
    padding = 8

    # 计算面板尺寸
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, tw)

    panel_w = max_w + padding * 2
    panel_h = line_height * len(lines) + padding * 2

    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # 文字
    y = padding + 16
    for line in lines:
        cv2.putText(img, line, (padding, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height

    return img
