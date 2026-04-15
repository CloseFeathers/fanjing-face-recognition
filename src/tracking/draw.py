"""
Tracking visualization — Draw tracking boxes with track_id/person_id coloring + HUD on frame.
"""

from __future__ import annotations

import colorsys
from typing import Dict, Optional

import cv2
import numpy as np

from .track import FrameTracks


def _track_color(track_id: int) -> tuple:
    """Generate stable BGR color based on track_id."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def _person_color(person_id: int) -> tuple:
    """Generate stable BGR color based on person_id (different from track color)."""
    # Use different hue offset
    hue = (person_id * 0.381966 + 0.15) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.95, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_tracks(
    image: np.ndarray,
    ft: FrameTracks,
    pipeline_fps: float,
    dropped_frames: int = 0,
    det_faces: int = 0,
    track_to_person: Optional[Dict[int, int]] = None,
    show_person: bool = True,
    show_similarity: bool = False,
    track_similarities: Optional[Dict[int, float]] = None,
    person_count: int = 0,
    person_identity_states: Optional[Dict[int, str]] = None,
    person_to_candidate: Optional[Dict[int, int]] = None,
    person_to_registered: Optional[Dict[int, int]] = None,
    mouth_states: Optional[Dict[int, str]] = None,
    show_hud: bool = True,
) -> np.ndarray:
    """
    Overlay tracking results and HUD on frame.

    Args:
        image: Original image
        ft: FrameTracks object
        pipeline_fps: Pipeline FPS
        dropped_frames: Dropped frame count
        det_faces: Detected face count
        track_to_person: track_id -> person_id mapping
        show_person: Whether to show person_id
        show_similarity: Whether to show similarity score
        track_similarities: track_id -> similarity mapping
        person_count: Current person total count
        person_identity_states: session_person_id -> identity_state (Module 5)
        person_to_candidate: session_person_id -> candidate_id (Module 5)

    Returns:
        Rendered image
    """
    img = image.copy()

    if track_to_person is None:
        track_to_person = {}
    if track_similarities is None:
        track_similarities = {}
    if person_identity_states is None:
        person_identity_states = {}
    if person_to_candidate is None:
        person_to_candidate = {}
    if person_to_registered is None:
        person_to_registered = {}
    if mouth_states is None:
        mouth_states = {}

    # ------------------------------------------------------------------
    # 1. Draw each track
    # ------------------------------------------------------------------
    for trk in ft.tracks:
        tid = trk["track_id"]
        bbox = trk["bbox_xyxy"]
        state = trk["state"]
        score = trk.get("det_score")
        pid = track_to_person.get(tid)

        if not show_hud:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]

        if pid is not None and show_person:
            color = _person_color(pid)
        else:
            color = _track_color(tid)

        thickness = 2 if state == "confirmed" else 1
        if state == "lost":
            _draw_dashed_rect(img, (x1, y1), (x2, y2), color, thickness)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if show_hud:
            label = f"T#{tid}"
            if show_person and pid is not None:
                label += f" P#{pid}"
                rid = person_to_registered.get(pid)
                if rid is not None:
                    label += f" R#{rid}"
                identity_state = person_identity_states.get(pid)
                if identity_state:
                    state_short = {
                        "KNOWN_STRONG": "KS",
                        "AMBIGUOUS": "AMB",
                        "UNKNOWN_STRONG": "US",
                    }.get(identity_state, identity_state[:3])
                    label += f" {state_short}"
                cid = person_to_candidate.get(pid)
                if cid is not None:
                    label += f" C#{cid}"
            credit = trk.get("face_valid_credit")
            if credit is not None and credit > 0:
                label += f" C{credit:.0f}"
            mouth = mouth_states.get(tid)
            if mouth == "speaking":
                label += " SPK"
            elif mouth == "not_speaking":
                label += " SIL"
            elif mouth == "occluded":
                label += " OCC"
            elif mouth == "self_occluded":
                label += " S-OCC"
            elif mouth == "unobservable":
                label += " ?"
            if show_similarity and tid in track_similarities:
                sim = track_similarities[tid]
                label += f" ({sim:.2f})"
            elif score is not None:
                label += f" {score:.2f}"

            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 2. HUD info panel (top-left)
    # ------------------------------------------------------------------
    if show_hud:
        lines = [
            f"source  : {ft.source_id}",
            f"frame   : {ft.frame_id}",
            f"ts(ms)  : {ft.timestamp_ms:.1f}",
            f"size    : {ft.width}x{ft.height}",
            f"FPS     : {pipeline_fps:.1f}",
            f"det(ms) : {ft.detect_time_ms:.1f}",
            f"trk(ms) : {ft.track_time_ms:.1f}",
            f"faces   : {det_faces}",
            f"tracks  : {ft.num_tracks}",
            f"persons : {person_count}",
            f"dropped : {dropped_frames}",
            f"detect  : {'Y' if ft.did_detect else 'N'}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness_t = 1
        line_h = 20
        pad = 8

        max_w = max(cv2.getTextSize(ln, font, font_scale, thickness_t)[0][0] for ln in lines)
        panel_w = max_w + pad * 2
        panel_h = line_h * len(lines) + pad * 2

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        y = pad + 14
        for line in lines:
            cv2.putText(img, line, (pad, y), font, font_scale,
                        (0, 255, 0), thickness_t, cv2.LINE_AA)
            y += line_h

    return img


def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash_len=8):
    """Draw dashed rectangle (for lost tracks)."""
    x1, y1 = pt1
    x2, y2 = pt2
    for start, end in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        _draw_dashed_line(img, start, end, color, thickness, dash_len)


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len):
    dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
    if dist < 1:
        return
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist
    i = 0
    while i < dist:
        s = (int(pt1[0] + dx * i), int(pt1[1] + dy * i))
        e_len = min(dash_len, dist - i)
        e = (int(pt1[0] + dx * (i + e_len)), int(pt1[1] + dy * (i + e_len)))
        cv2.line(img, s, e, color, thickness)
        i += dash_len * 2
