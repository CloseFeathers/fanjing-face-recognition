"""
Module 2: BoT-SORT Tracking (无 ReID)
将逐帧 detections 关联为连续轨迹，输出稳定的 track_id。
"""

from .bot_sort import BoTSORTConfig, BoTSORTTracker
from .draw import draw_tracks
from .track import FrameTracks, STrack, TrackState

__all__ = [
    "STrack", "TrackState", "FrameTracks",
    "BoTSORTTracker", "BoTSORTConfig",
    "draw_tracks",
]
