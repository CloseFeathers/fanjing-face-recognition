#!/usr/bin/env python3
"""
Module 1 Acceptance Entry — Face detection preview + JSONL logging.

Usage:
  # Download model first (initial setup)
  python scripts/download_model.py

  # Camera mode (press q to quit)
  python run_detection.py --camera 0

  # Video file mode - run as fast as possible
  python run_detection.py --video path/to/video.mp4

  # Video file mode - play at original frame rate
  python run_detection.py --video path/to/video.mp4 --realtime

  # With JSONL logging
  python run_detection.py --camera 0 --log
  python run_detection.py --video path/to/video.mp4 --log

  # Specify model file / threshold / detection size / GPU
  python run_detection.py --camera 0 --model models/det_10g.onnx --det-thresh 0.4 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Windows terminal UTF-8 output protection
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import cv2

from src.ingestion.camera_source import CameraSource
from src.ingestion.video_source import VideoSource
from src.detectors.scrfd_detector import SCRFDDetector
from src.detectors.draw import draw_detections


WINDOW_NAME = "Detection Preview"


# ======================================================================
# JSONL Logger
# ======================================================================

class DetectionLogger:
    """Write FrameDetections as JSONL to file."""

    def __init__(self, path: str = "output/detections.jsonl") -> None:
        self._path = Path(path)
        self._fp = None

    def open(self) -> "DetectionLogger":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._path, "w", encoding="utf-8")
        return self

    def log(self, dets_dict: dict) -> None:
        if self._fp is not None:
            self._fp.write(json.dumps(dets_dict, ensure_ascii=False) + "\n")
            self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> "DetectionLogger":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


# ======================================================================
# FPS Counter
# ======================================================================

class FPSCounter:
    """1-second sliding window FPS counter."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._count = 0
        self._fps = 0.0

    def tick(self) -> float:
        self._count += 1
        now = time.monotonic()
        elapsed = now - self._start
        if elapsed >= 1.0:
            self._fps = self._count / elapsed
            self._count = 0
            self._start = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ======================================================================
# Camera Mode
# ======================================================================

def run_camera(
    device: int,
    detector: SCRFDDetector,
    logger: DetectionLogger | None,
) -> None:
    print(f"[Detection] Camera mode  device={device}")
    print("[Detection] Press q to quit")

    fps_counter = FPSCounter()
    last_dets = None

    with CameraSource(device=device) as src:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            frame = src.read(timeout=5.0)
            if frame is None:
                print("[Detection] Read timeout, retrying...")
                continue

            dets = detector.detect(frame)
            last_dets = dets
            pipeline_fps = fps_counter.tick()

            display = draw_detections(
                frame.image, dets, pipeline_fps, src.dropped_frames
            )
            cv2.imshow(WINDOW_NAME, display)

            if logger:
                logger.log(dets.to_dict())

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    if last_dets:
        print(f"[Detection] Done. last frame_id={last_dets.frame_id}, "
              f"dropped={src.dropped_frames}")


# ======================================================================
# Video File Mode
# ======================================================================

def run_video(
    path: str,
    realtime: bool,
    detector: SCRFDDetector,
    logger: DetectionLogger | None,
) -> None:
    print(f"[Detection] Video file mode  path={path}  realtime={realtime}")
    print("[Detection] Press q to quit")

    fps_counter = FPSCounter()

    with VideoSource(path=path, realtime=realtime) as src:
        print(f"[Detection] Video fps={src.video_fps:.2f}, "
              f"total_frames={src.total_frames}")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        last_dets = None
        while True:
            frame = src.read()
            if frame is None:
                print("[Detection] Video playback finished.")
                break

            dets = detector.detect(frame)
            last_dets = dets
            pipeline_fps = fps_counter.tick()

            display = draw_detections(
                frame.image, dets, pipeline_fps, src.dropped_frames
            )
            cv2.imshow(WINDOW_NAME, display)

            if logger:
                logger.log(dets.to_dict())

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Detection] User aborted.")
                break

    cv2.destroyAllWindows()
    if last_dets:
        print(f"[Detection] Done. last frame_id={last_dets.frame_id}, "
              f"last ts={last_dets.timestamp_ms:.1f}ms")


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 1: Face Detection preview & logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Input Source ----
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--camera", type=int, metavar="DEVICE",
                     help="Camera device index, e.g. 0")
    grp.add_argument("--video", type=str, metavar="PATH",
                     help="Video file path")
    parser.add_argument("--realtime", action="store_true",
                        help="Play video at original FPS (default: fast-forward)")

    # ---- Detector Parameters ----
    parser.add_argument("--model", type=str, default="models/det_10g.onnx",
                        help="SCRFD ONNX model path (default: models/det_10g.onnx)")
    parser.add_argument("--det-size", type=int, default=640,
                        help="Detection input size (default: 640)")
    parser.add_argument("--det-thresh", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--nms-thresh", type=float, default=0.4,
                        help="NMS IoU threshold (default: 0.4)")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU id, -1 = CPU only (default: -1)")

    # ---- Logging ----
    parser.add_argument("--log", action="store_true",
                        help="Write detections JSONL to output/detections.jsonl")
    parser.add_argument("--log-path", type=str,
                        default="output/detections.jsonl",
                        help="JSONL log path (default: output/detections.jsonl)")

    args = parser.parse_args()

    # ---- Load Detector ----
    det_size = (args.det_size, args.det_size)
    detector = SCRFDDetector(
        model_path=args.model,
        det_size=det_size,
        det_thresh=args.det_thresh,
        nms_thresh=args.nms_thresh,
        gpu_id=args.gpu,
    )
    detector.load()

    # ---- Logging ----
    logger = None
    if args.log:
        logger = DetectionLogger(path=args.log_path)
        logger.open()
        print(f"[Detection] JSONL -> {args.log_path}")

    try:
        if args.camera is not None:
            run_camera(args.camera, detector, logger)
        else:
            run_video(args.video, args.realtime, detector, logger)
    except KeyboardInterrupt:
        print("\n[Detection] Ctrl+C")
    finally:
        if logger:
            logger.close()
            print("[Detection] JSONL saved.")


if __name__ == "__main__":
    main()
