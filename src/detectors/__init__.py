"""
Module 1: Face Detection
从 Ingestion 输出的 Frame 中检测人脸，输出统一的 Detection 结果。
使用 SCRFD (ONNX Runtime) 作为检测底座，不依赖 insightface 包。
"""

from .detection import FaceDetection, FrameDetections
from .draw import draw_detections
from .scrfd_detector import SCRFDDetector

__all__ = ["FaceDetection", "FrameDetections", "SCRFDDetector", "draw_detections"]
