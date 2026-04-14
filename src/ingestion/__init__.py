"""
Module 0: Video Ingestion & Timestamp Layer
统一摄像头 / 视频文件两种输入源，对外输出标准 Frame 对象。
"""

from .camera_source import CameraSource
from .frame import Frame
from .logger import FrameLogger
from .overlay import draw_overlay
from .video_source import VideoSource

__all__ = ["Frame", "CameraSource", "VideoSource", "draw_overlay", "FrameLogger"]
