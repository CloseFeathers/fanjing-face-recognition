"""
Module 1: Face Detection
Detect faces from Ingestion output Frame, output unified Detection results.
Uses SCRFD (ONNX Runtime) as detection base, no dependency on insightface package.
"""

from .detection import FaceDetection, FrameDetections
from .draw import draw_detections
from .scrfd_detector import SCRFDDetector

__all__ = ["FaceDetection", "FrameDetections", "SCRFDDetector", "draw_detections"]
