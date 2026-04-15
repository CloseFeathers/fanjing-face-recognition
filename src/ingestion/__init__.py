"""
Module 0: Video Ingestion & Timestamp Layer

Unifies camera and video file input sources, outputs standard Frame objects.
"""

from .camera_source import CameraSource
from .frame import Frame
from .logger import FrameLogger
from .overlay import draw_overlay
from .video_source import VideoSource

__all__ = ["Frame", "CameraSource", "VideoSource", "draw_overlay", "FrameLogger"]
