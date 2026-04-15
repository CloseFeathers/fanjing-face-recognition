"""
Module 2: BoT-SORT Tracking (without ReID)
Associate per-frame detections into continuous trajectories, output stable track_id.
"""

from .bot_sort import BoTSORTConfig, BoTSORTTracker
from .draw import draw_tracks
from .track import FrameTracks, STrack, TrackState

__all__ = [
    "STrack", "TrackState", "FrameTracks",
    "BoTSORTTracker", "BoTSORTConfig",
    "draw_tracks",
]
